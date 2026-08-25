"""Inference-only SkyFare product demo."""

from __future__ import annotations

import traceback
from datetime import datetime, timedelta

import reflex as rx

from .service import AIRLINE_NAMES, AIRPORT_NAMES, service

INK = "#182126"
MUTED = "#637077"
LINE = "#DDE4E7"
PAGE = "#F4F7F7"
PANEL = "#FFFFFF"
TEAL = "#176B6B"
TEAL_SOFT = "#E7F3F1"
BLUE = "#3F6B8F"
BLUE_SOFT = "#EAF1F6"
CORAL = "#D85B4B"
GREEN = "#2E7D5B"
AMBER = "#A66A13"
MONO = "'SFMono-Regular', Consolas, monospace"
NAV_ITEMS = (
    "Overview",
    "Data Analysis",
    "Feature Engineering",
    "Modelling",
    "Evaluation",
    "Limitations",
    "Interactive Demo",
)

def airport_label(code: str) -> str:
    return f"{code} \u00b7 {AIRPORT_NAMES.get(code, code)}"


class DemoState(rx.State):
    active_tab: str = "Overview"
    ready: bool = False
    loading: bool = False
    searching: bool = False
    error: str = ""
    model_version: str = ""
    cutoff: str = ""
    load_time: str = ""
    session_dates: list[str] = []
    flight_dates: list[str] = []
    routes: list[str] = []
    airlines: list[str] = []
    route_airlines: dict[str, list[str]] = {}
    origin_options: list[dict[str, str]] = []
    destination_options: list[dict[str, str]] = []
    airline_options: list[dict[str, str]] = []

    session_date: str = ""
    session_label: str = "AM"
    origin: str = "SGN"
    destination: str = "HAN"
    flight_date: str = ""
    airline: str = ""

    results: list[dict] = []
    result_count: int = 0
    dud: int = 0
    on_grid: bool = True
    query_time: str = ""
    cheapest_price: str = ""
    cheapest_airline: str = ""
    cheapest_logo: str = ""
    cheapest_logo_alt: str = ""
    cheapest_time: str = ""
    expanded_slot: str = ""

    @rx.event
    def set_active_tab(self, value: str):
        self.active_tab = value

    @rx.event
    def initialize(self):
        if self.ready or self.loading:
            return
        self.loading = True
        self.error = ""
        yield
        try:
            catalog = service.initialize()
            self.session_dates = catalog["dates"]
            self.routes = catalog["routes"]
            self.airlines = catalog["airlines"]
            self.route_airlines = catalog["route_airlines"]
            self.cutoff = catalog["cutoff"]
            self.model_version = catalog["model_version"]
            self.load_time = f"{catalog['load_seconds']:.1f}s"
            self.session_date = self.session_dates[-1]
            session_day = datetime.fromisoformat(self.session_date).date()
            self.flight_dates = [
                (session_day + timedelta(days=dud)).isoformat()
                for dud in range(1, 61)
            ]
            self.flight_date = self.flight_dates[29]
            if "SGN-HAN" not in self.routes and self.routes:
                self.origin, self.destination = self.routes[0].split("-")
            self._sync_route_options("origin")
            self.ready = True
        except Exception as exc:
            self.error = f"Model load failed: {exc}"
        finally:
            self.loading = False

    @rx.event
    def set_session_date(self, value: str):
        self.session_date = value
        if value:
            session_day = datetime.fromisoformat(value).date()
            self.flight_dates = [
                (session_day + timedelta(days=dud)).isoformat()
                for dud in range(1, 61)
            ]
            self.flight_date = self.flight_dates[29]
        self.results = []

    @rx.event
    def set_session_label(self, value: str | list[str]):
        self.session_label = value[0] if isinstance(value, list) else value
        self.results = []

    @rx.event
    def set_origin(self, value: str):
        self.origin = value
        valid_destinations = self._destinations_for(value)
        if self.destination not in valid_destinations:
            self.destination = (
                valid_destinations[0] if valid_destinations else ""
            )
        self._sync_route_options("origin")
        self.results = []

    @rx.event
    def set_destination(self, value: str):
        self.destination = value
        valid_origins = self._origins_for(value)
        if self.origin not in valid_origins:
            self.origin = valid_origins[0] if valid_origins else ""
        self._sync_route_options("destination")
        self.results = []

    @rx.event
    def set_flight_date(self, value: str):
        self.flight_date = value
        self.results = []

    @rx.event
    def set_airline(self, value: str):
        selected = "" if value == "ALL" else value
        supported = self.route_airlines.get(self.route, [])
        self.airline = selected if selected in supported else ""
        self.results = []

    def _origins_for(self, destination: str) -> list[str]:
        return sorted(
            {
                route.split("-", 1)[0]
                for route in self.routes
                if route.endswith(f"-{destination}")
            }
        )

    def _destinations_for(self, origin: str) -> list[str]:
        return sorted(
            {
                route.split("-", 1)[1]
                for route in self.routes
                if route.startswith(f"{origin}-")
            }
        )

    @staticmethod
    def _airport_options(codes: list[str]) -> list[dict[str, str]]:
        return [
            {"value": code, "label": airport_label(code)}
            for code in codes
        ]

    def _sync_route_options(self, driver: str) -> None:
        all_origins = sorted(
            {route.split("-", 1)[0] for route in self.routes}
        )
        all_destinations = sorted(
            {route.split("-", 1)[1] for route in self.routes}
        )
        if driver == "destination":
            origin_codes = self._origins_for(self.destination)
            destination_codes = all_destinations
        else:
            origin_codes = all_origins
            destination_codes = self._destinations_for(self.origin)
        self.origin_options = self._airport_options(origin_codes)
        self.destination_options = self._airport_options(destination_codes)
        supported = self.route_airlines.get(self.route, [])
        if self.airline not in supported:
            self.airline = ""
        self.airline_options = [
            {"value": "ALL", "label": "All airlines"},
            *[
                {
                    "value": code,
                    "label": f"{AIRLINE_NAMES.get(code, code)} ({code})",
                }
                for code in supported
            ],
        ]

    @rx.event
    def toggle_slot(self, slot_id: str):
        self.expanded_slot = "" if self.expanded_slot == slot_id else slot_id

    @rx.var
    def route(self) -> str:
        if not self.origin or not self.destination:
            return ""
        return f"{self.origin}-{self.destination}"

    @rx.var
    def can_search(self) -> bool:
        return bool(
            self.ready
            and self.route in self.routes
            and (
                not self.airline
                or self.airline in self.route_airlines.get(self.route, [])
            )
            and self.session_date
            and self.flight_date
            and not self.searching
        )

    @rx.var
    def cheapest_context(self) -> str:
        return self.cheapest_time

    @rx.event
    def run_search(self):
        self.error = ""
        self.results = []
        self.expanded_slot = ""
        self.searching = True
        yield
        try:
            payload = service.search(
                session_date=self.session_date,
                session_label=self.session_label,
                route=self.route,
                flight_date=self.flight_date,
                airline=self.airline,
            )
            self.results = payload["rows"]
            self.result_count = payload["count"]
            self.dud = payload["dud"]
            self.on_grid = payload["on_grid"]
            self.query_time = f"{payload['query_seconds']:.2f}s"
            self.cheapest_price = payload["cheapest_price"]
            self.cheapest_airline = payload["cheapest_airline"]
            self.cheapest_logo = payload["cheapest_logo"]
            self.cheapest_logo_alt = payload["cheapest_logo_alt"]
            self.cheapest_time = payload["cheapest_time"]
        except Exception as exc:
            traceback.print_exc()
            self.error = str(exc)
        finally:
            self.searching = False


def field(label: str, control: rx.Component) -> rx.Component:
    return rx.vstack(
        rx.text(label, size="1", weight="bold", color=MUTED),
        control,
        spacing="2",
        align="start",
        width="100%",
    )


def select_field(
    value: rx.Var,
    on_change,
    placeholder: str,
    items: list[tuple[str, str]],
) -> rx.Component:
    return rx.el.select(
        *[
            rx.el.option(label, value=code)
            for code, label in items
        ],
        value=value,
        on_change=on_change,
        aria_label=placeholder,
        class_name="query-native-select",
    )


def linked_select_field(
    value: rx.Var,
    on_change,
    placeholder: str,
    items: rx.Var,
) -> rx.Component:
    return rx.el.select(
        rx.foreach(
            items,
            lambda item: rx.el.option(
                item["label"],
                value=item["value"],
            ),
        ),
        value=value,
        on_change=on_change,
        aria_label=placeholder,
        class_name="query-native-select",
    )


def dynamic_date_select() -> rx.Component:
    return rx.el.select(
        rx.foreach(
            DemoState.session_dates,
            lambda value: rx.el.option(value, value=value),
        ),
        value=DemoState.session_date,
        on_change=DemoState.set_session_date,
        aria_label="Booking date",
        class_name="query-native-select query-date-select",
    )


def dynamic_flight_date_select() -> rx.Component:
    return rx.el.select(
        rx.foreach(
            DemoState.flight_dates,
            lambda value: rx.el.option(value, value=value),
        ),
        value=DemoState.flight_date,
        on_change=DemoState.set_flight_date,
        aria_label="Flight date",
        class_name="query-native-select query-date-select",
    )


def calendar_control(control: rx.Component) -> rx.Component:
    return rx.box(
        control,
        rx.icon(
            "calendar-days",
            size=17,
            color=TEAL,
            position="absolute",
            left="0.85rem",
            top="50%",
            transform="translateY(-50%)",
            pointer_events="none",
            z_index="1",
        ),
        position="relative",
        width="100%",
    )


def navigation_item(label: str) -> rx.Component:
    active = DemoState.active_tab == label
    return rx.button(
        label,
        on_click=DemoState.set_active_tab(label),
        variant="ghost",
        color=rx.cond(active, TEAL, MUTED),
        background=rx.cond(active, TEAL_SOFT, "transparent"),
        border_radius="6px",
        height="36px",
        padding_x="0.9rem",
        margin="0",
        font_size="0.84rem",
        font_weight=rx.cond(active, "700", "500"),
        white_space="nowrap",
        cursor="pointer",
        _hover={
            "background": TEAL_SOFT,
            "color": TEAL,
        },
    )


def model_header() -> rx.Component:
    return rx.box(
        rx.box(
            rx.hstack(
                rx.hstack(
                    rx.box(
                        rx.icon("plane", size=21, color="white"),
                        width="38px",
                        height="38px",
                        display="flex",
                        align_items="center",
                        justify_content="center",
                        background=TEAL,
                        border_radius="7px",
                    ),
                    rx.vstack(
                        rx.heading(
                            "SkyFare",
                            size="5",
                            color=INK,
                            letter_spacing="0",
                        ),
                        rx.text(
                            "Fare intelligence",
                            size="1",
                            color=MUTED,
                        ),
                        spacing="0",
                        align="start",
                    ),
                    spacing="3",
                    align="center",
                ),
                rx.spacer(),
                rx.hstack(
                    rx.cond(
                        DemoState.loading,
                        rx.hstack(
                            rx.spinner(size="1"),
                            rx.text("Loading ensemble", size="1"),
                            spacing="2",
                        ),
                        rx.hstack(
                            rx.box(
                                width="8px",
                                height="8px",
                                border_radius="50%",
                                background=rx.cond(
                                    DemoState.ready, GREEN, CORAL
                                ),
                            ),
                            rx.text(
                                rx.cond(
                                    DemoState.ready,
                                    "Model ready",
                                    "Model unavailable",
                                ),
                                size="1",
                                weight="bold",
                                color=INK,
                            ),
                            spacing="2",
                            align="center",
                        ),
                    ),
                    rx.badge(
                        "150-day production refit",
                        variant="soft",
                        color_scheme="teal",
                    ),
                    spacing="3",
                    align="center",
                ),
                width="100%",
                align="center",
            ),
            rx.hstack(
                *[navigation_item(label) for label in NAV_ITEMS],
                spacing="3",
                align="center",
                width="100%",
                margin_top="0.65rem",
            ),
            max_width="1240px",
            width="100%",
            margin="0 auto",
            padding_x=rx.breakpoints(
                initial="1rem", md="1.5rem", lg="2rem"
            ),
        ),
        background=PANEL,
        border_bottom=f"1px solid {LINE}",
        padding_y="0.75rem",
        width="100%",
        position="sticky",
        top="0",
        z_index="20",
    )


def overview_stat(value: str, label: str, icon: str) -> rx.Component:
    return rx.vstack(
        rx.icon(icon, size=18, color=TEAL),
        rx.text(
            value,
            size="6",
            color=INK,
            weight="bold",
            font_family=MONO,
        ),
        rx.text(
            label,
            size="1",
            color=MUTED,
            text_align="center",
            line_height="1.35",
        ),
        spacing="2",
        align="center",
        min_height="118px",
        justify="center",
        padding="1rem 0.75rem",
        border_right=f"1px solid {LINE}",
    )


def overview_scope_chip(label: str, tint: str = TEAL_SOFT) -> rx.Component:
    return rx.text(
        label,
        size="1",
        color=INK,
        weight="medium",
        font_family=MONO,
        background=tint,
        border=f"1px solid {LINE}",
        border_radius="5px",
        padding="0.42rem 0.65rem",
        white_space="nowrap",
    )


def overview_scope_group(
    eyebrow: str,
    title: str,
    description: str,
    labels: list[str],
    tint: str = TEAL_SOFT,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(eyebrow, size="1", color=TEAL, weight="bold"),
            rx.spacer(),
            rx.text(
                f"{len(labels)} ITEMS",
                size="1",
                color=MUTED,
                weight="bold",
                font_family=MONO,
            ),
            width="100%",
            align="center",
        ),
        rx.heading(title, size="3", color=INK, letter_spacing="0"),
        rx.text(
            description,
            size="1",
            color=MUTED,
            line_height="1.45",
        ),
        rx.flex(
            *[overview_scope_chip(label, tint) for label in labels],
            wrap="wrap",
            gap="0.5rem",
            width="100%",
        ),
        spacing="2",
        align="start",
        width="100%",
        padding="1rem",
        border_top=f"2px solid {TEAL}",
        background=PANEL,
    )


def task_summary(
    icon: str,
    eyebrow: str,
    title: str,
    description: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon(icon, size=22, color=accent),
            width="44px",
            height="44px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=tint,
            border_radius="7px",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.text(
                eyebrow,
                size="1",
                color=accent,
                weight="bold",
            ),
            rx.heading(
                title,
                size="4",
                color=INK,
                letter_spacing="0",
            ),
            rx.text(
                description,
                size="2",
                color=MUTED,
                line_height="1.55",
                max_width="500px",
            ),
            spacing="1",
            align="start",
        ),
        spacing="4",
        align="start",
        width="100%",
        padding="1.25rem",
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
    )


def lifecycle_step(
    number: str,
    title: str,
    detail: str,
    final: bool = False,
) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.text(
                number,
                size="1",
                color="white",
                weight="bold",
                font_family=MONO,
            ),
            width="28px",
            height="28px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=TEAL if not final else BLUE,
            border_radius="50%",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.text(title, size="2", color=INK, weight="bold"),
            rx.text(
                detail,
                size="1",
                color=MUTED,
                line_height="1.4",
            ),
            spacing="0",
            align="start",
        ),
        spacing="3",
        align="start",
        min_height="72px",
        padding="0.8rem",
        background=rx.cond(final, BLUE_SOFT, PANEL),
        border_left=f"1px solid {LINE}",
    )


def overview_page() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text(
                        "PROJECT OVERVIEW",
                        size="1",
                        color=TEAL,
                        weight="bold",
                    ),
                    rx.heading(
                        "Flight fare intelligence built for temporal change",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                        line_height="1.12",
                        max_width="760px",
                    ),
                    rx.text(
                        "SkyFare estimates schedule-slot fares for future "
                        "departure dates, ranks affordable options, and adds "
                        "price-drop evidence to support booking decisions.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                        max_width="760px",
                    ),
                    spacing="3",
                    align="start",
                ),
                rx.box(
                    rx.vstack(
                        rx.text(
                            "TEMPORAL EVIDENCE",
                            size="1",
                            color=BLUE,
                            weight="bold",
                        ),
                        rx.hstack(
                            rx.vstack(
                                rx.text(
                                    "6",
                                    size="7",
                                    color=INK,
                                    weight="bold",
                                    font_family=MONO,
                                ),
                                rx.text(
                                    "development folds",
                                    size="1",
                                    color=MUTED,
                                ),
                                spacing="0",
                                align="start",
                            ),
                            rx.box(
                                width="1px",
                                height="48px",
                                background=LINE,
                            ),
                            rx.vstack(
                                rx.text(
                                    "2",
                                    size="7",
                                    color=INK,
                                    weight="bold",
                                    font_family=MONO,
                                ),
                                rx.text(
                                    "locked validations",
                                    size="1",
                                    color=MUTED,
                                ),
                                spacing="0",
                                align="start",
                            ),
                            spacing="5",
                            align="center",
                        ),
                        rx.text(
                            "One later temporal period remained reserved for "
                            "final evaluation.",
                            size="1",
                            color=MUTED,
                            line_height="1.45",
                        ),
                        spacing="3",
                        align="start",
                    ),
                    background=BLUE_SOFT,
                    border_left=f"4px solid {BLUE}",
                    padding="1.25rem",
                    min_height="180px",
                    display="flex",
                    align_items="center",
                ),
                grid_template_columns="minmax(0, 1.7fr) minmax(300px, 0.8fr)",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border_bottom=f"1px solid {LINE}",
            padding="2.25rem 2rem",
            width="100%",
        ),
        rx.grid(
            task_summary(
                "chart-no-axes-combined",
                "REGRESSION",
                "Estimate and rank future departure fares",
                "Predicts the fare of each eligible schedule slot at the "
                "selected booking session, across all airlines or within one "
                "airline.",
                TEAL,
                TEAL_SOFT,
            ),
            task_summary(
                "percent",
                "CLASSIFICATION",
                "Estimate material price-drop probability",
                "Estimates whether fare may fall by at least 5% at the next "
                "canonical booking window and supports BUY or WAIT guidance.",
                BLUE,
                BLUE_SOFT,
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.text(
                            "DATA SCOPE",
                            size="1",
                            color=TEAL,
                            weight="bold",
                        ),
                        rx.heading(
                            "A consistent standard-fare population",
                            size="5",
                            color=INK,
                            letter_spacing="0",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    rx.spacer(),
                    rx.text(
                        "Google Flights and Trip.com standard periods",
                        size="1",
                        color=MUTED,
                    ),
                    width="100%",
                    align="end",
                ),
                rx.grid(
                    overview_stat("150", "study days", "calendar-range"),
                    overview_stat("20", "directional routes", "route"),
                    overview_stat("5", "airlines", "plane"),
                    overview_stat("72", "route-airline pairs", "git-branch"),
                    overview_stat("11", "booking windows", "calendar-check-2"),
                    columns="5",
                    spacing="0",
                    width="100%",
                    border=f"1px solid {LINE}",
                    border_radius="8px",
                    overflow="hidden",
                    background=PANEL,
                ),
                rx.grid(
                    overview_stat("ONE-WAY", "journey", "arrow-right"),
                    overview_stat("1 ADULT", "passenger", "user-round"),
                    overview_stat("ECONOMY", "cabin class", "armchair"),
                    columns="3",
                    spacing="0",
                    width="100%",
                    border=f"1px solid {LINE}",
                    border_radius="8px",
                    overflow="hidden",
                    background=PANEL,
                ),
                overview_scope_group(
                    "ROUTE SCOPE",
                    "Twenty directional domestic routes",
                    "Opposite directions remain separate markets. Airport codes: "
                    "SGN Ho Chi Minh City, HAN Hanoi, PQC Phu Quoc, DAD Da Nang, "
                    "CXR Nha Trang, HPH Hai Phong and VCA Can Tho.",
                    [
                        "SGN \u2192 HAN", "HAN \u2192 SGN",
                        "SGN \u2192 PQC", "PQC \u2192 SGN",
                        "HAN \u2192 PQC", "PQC \u2192 HAN",
                        "DAD \u2192 PQC", "PQC \u2192 DAD",
                        "SGN \u2192 DAD", "DAD \u2192 SGN",
                        "HAN \u2192 DAD", "DAD \u2192 HAN",
                        "SGN \u2192 CXR", "CXR \u2192 SGN",
                        "HAN \u2192 CXR", "CXR \u2192 HAN",
                        "SGN \u2192 HPH", "HPH \u2192 SGN",
                        "HAN \u2192 VCA", "VCA \u2192 HAN",
                    ],
                ),
                rx.grid(
                    overview_scope_group(
                        "AIRLINE SCOPE",
                        "Five domestic airlines",
                        "Only observed standard Economy fares enter the common contract.",
                        [
                            "VN \u00b7 Vietnam Airlines",
                            "VJ \u00b7 VietJet Air",
                            "QH \u00b7 Bamboo Airways",
                            "VU \u00b7 Vietravel Airlines",
                            "9G \u00b7 Sun Phu Quoc Airways",
                        ],
                    ),
                    overview_scope_group(
                        "BOOKING-WINDOW SCOPE",
                        "Eleven canonical observation windows",
                        "DUD means days until departure; observations become denser near the flight date.",
                        [
                            "DUD 60", "DUD 45", "DUD 30", "DUD 21",
                            "DUD 14", "DUD 10", "DUD 7", "DUD 5",
                            "DUD 3", "DUD 2", "DUD 1",
                        ],
                        BLUE_SOFT,
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                    align_items="stretch",
                ),
                spacing="4",
                width="100%",
            ),
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "DEVELOPMENT LIFECYCLE",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                rx.heading(
                    "From collected fares to a frozen deployment system",
                    size="5",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.grid(
                    lifecycle_step(
                        "01",
                        "Collect",
                        "Capture scheduled fares over time.",
                    ),
                    lifecycle_step(
                        "02",
                        "Standardise",
                        "Create one comparable fare population.",
                    ),
                    lifecycle_step(
                        "03",
                        "Engineer",
                        "Build legal task-specific evidence.",
                    ),
                    lifecycle_step(
                        "04",
                        "Develop",
                        "Compare models through temporal folds.",
                    ),
                    lifecycle_step(
                        "05",
                        "Validate",
                        "Check frozen systems on later windows.",
                    ),
                    lifecycle_step(
                        "06",
                        "Evaluate",
                        "Measure once on final temporal data.",
                    ),
                    lifecycle_step(
                        "07",
                        "Deploy",
                        "Refit locked recipes for inference.",
                        final=True,
                    ),
                    columns="7",
                    spacing="0",
                    width="100%",
                    margin_top="0.75rem",
                    border_top=f"1px solid {LINE}",
                    border_bottom=f"1px solid {LINE}",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            background="#F5F9F8",
            padding="1.5rem",
            width="100%",
        ),
        spacing="6",
        width="100%",
    )


def evidence_stat(value: str, label: str, accent: str = TEAL) -> rx.Component:
    return rx.vstack(
        rx.text(
            value,
            size="5",
            color=accent,
            weight="bold",
            font_family=MONO,
        ),
        rx.text(
            label,
            size="1",
            color=MUTED,
            line_height="1.35",
            text_align="center",
        ),
        spacing="0",
        align="center",
        min_width="118px",
    )


def source_definition(
    icon: str,
    name: str,
    role: str,
    description: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon(icon, size=20, color=accent),
            width="40px",
            height="40px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=tint,
            border_radius="7px",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.hstack(
                rx.text(name, size="2", color=INK, weight="bold"),
                rx.badge(
                    role,
                    variant="soft",
                    color_scheme="teal" if accent == TEAL else "blue",
                ),
                spacing="2",
                align="center",
            ),
            rx.text(
                description,
                size="1",
                color=MUTED,
                line_height="1.5",
            ),
            spacing="1",
            align="start",
        ),
        spacing="3",
        align="start",
        padding="1rem",
        border_left=f"3px solid {accent}",
        background=PANEL,
        min_height="118px",
        width="100%",
    )


def figure_panel(
    number: str,
    title: str,
    description: str,
    image: str,
    alt: str,
    takeaway: str,
    stats: list[tuple[str, str]],
    image_class: str = "",
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text(
                        f"FIGURE {number}",
                        size="1",
                        color=TEAL,
                        weight="bold",
                    ),
                    rx.heading(
                        title,
                        size="5",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        description,
                        size="2",
                        color=MUTED,
                        line_height="1.55",
                        max_width="790px",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                rx.hstack(
                    *[
                        evidence_stat(value, label, BLUE)
                        for value, label in stats
                    ],
                    spacing="5",
                    align="center",
                ),
                width="100%",
                align="center",
            ),
            rx.box(
                rx.image(
                    src=image,
                    alt=alt,
                    width="100%",
                    height="auto",
                    class_name=image_class,
                ),
                width="100%",
                overflow="hidden",
                border_top=f"1px solid {LINE}",
                border_bottom=f"1px solid {LINE}",
                padding_y="0.75rem",
            ),
            rx.hstack(
                rx.icon("lightbulb", size=17, color=TEAL),
                rx.text(
                    takeaway,
                    size="1",
                    color=INK,
                    line_height="1.5",
                ),
                spacing="2",
                align="start",
                width="100%",
                background=TEAL_SOFT,
                padding="0.75rem 0.9rem",
                border_radius="6px",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="1.25rem",
        width="100%",
    )


def data_analysis_page() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text(
                        "DATA ANALYSIS",
                        size="1",
                        color=TEAL,
                        weight="bold",
                    ),
                    rx.heading(
                        "Evidence before modelling",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        "Six checks establish whether collected fares are "
                        "continuous, comparable, representative, and "
                        "temporally informative.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                        max_width="720px",
                    ),
                    spacing="2",
                    align="start",
                ),
                rx.hstack(
                    evidence_stat("923,855", "development observations"),
                    rx.box(width="1px", height="58px", background=LINE),
                    evidence_stat("128", "development days", BLUE),
                    spacing="5",
                    align="center",
                    justify="end",
                ),
                grid_template_columns="minmax(0, 1.6fr) minmax(330px, 0.8fr)",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border_bottom=f"1px solid {LINE}",
            padding="2rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "COLLECTION ARCHITECTURE",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                rx.heading(
                    "Two sources, two acquisition paths",
                    size="5",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.text(
                    "Google Flights observations were collected through the "
                    "fli library. Trip.com observations were collected through "
                    "Playwright browser automation, with Camoufox providing "
                    "browser-environment compatibility and access-state reporting. Both streams "
                    "follow the same standard fare contract.",
                    size="2",
                    color=MUTED,
                    line_height="1.5",
                    max_width="800px",
                ),
                rx.grid(
                    source_definition(
                        "scan-search",
                        "Google Flights",
                        "fli library",
                        "Programmatic observations collected through "
                        "punitarani/fli for historical evidence and paired "
                        "source-compatibility checks.",
                        BLUE,
                        BLUE_SOFT,
                    ),
                    source_definition(
                        "mouse-pointer-click",
                        "Trip.com",
                        "Playwright + Camoufox",
                        "Playwright controls navigation, page interaction, and "
                        "DOM extraction; Camoufox supplies the resilient browser "
                        "environment, rate-limited retries, and explicit "
                        "failure reporting for daily fare evidence.",
                        TEAL,
                        TEAL_SOFT,
                    ),
                    columns="2",
                    spacing="3",
                    width="100%",
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            background="#F5F9F8",
            border=f"1px solid {LINE}",
            padding="1.25rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "POINT-IN-TIME DATA OPERATIONS",
                    size="1",
                    color=BLUE,
                    weight="bold",
                ),
                rx.image(
                    src=(
                        "/data_analysis/skyfare_live_postgresql_erd.svg"
                        "?v=20260804a"
                    ),
                    alt="SkyFare Live PostgreSQL data model",
                    width="100%",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            padding="1.25rem",
            width="100%",
        ),
        figure_panel(
            "01",
            "Development collection continuity",
            "Daily standard-fare volume across the development-history period "
            "reveals the Fli-to-Trip transition and collection gaps.",
            "/data_analysis/collection_coverage.png?v=20260730c",
            "Daily standard-fare observations across 128 observed development-history days",
            "Two uncollected calendar dates remain explicit; they are not "
            "filled or represented as observed sessions.",
            [("128", "observed history days"), ("2", "uncollected dates")],
        ),
        figure_panel(
            "02",
            "Development source compatibility",
            "Exact route, airline, flight date, departure minute, booking "
            "window, and session matches compare fli and Trip listings.",
            "/data_analysis/source_agreement.png?v=20260730c",
            "Paired fli and Trip price agreement with equality line and relative differences",
            "Agreement supports early-period standardisation. It does not "
            "claim identical sources or unchanged later-period semantics.",
            [("150k", "matched observations"), ("93.62%", "within \u00b15%")],
        ),
        figure_panel(
            "03",
            "Route-airline coverage",
            "Cell intensity shows observation support for each market pair; "
            "blank cells represent schedules absent from collected evidence.",
            "/data_analysis/route_airline_coverage.png?v=20260730c",
            "Route by airline observation support matrix",
            "The candidate system must respect 72 observed route-airline "
            "pairs rather than assume every airline serves every route.",
            [("20", "routes"), ("72", "supported pairs")],
            "coverage-matrix-image",
        ),
        figure_panel(
            "04",
            "Airline observation density",
            "Counts describe collected schedule availability and volume, not "
            "airline importance or data quality.",
            "/data_analysis/airline_density.png?v=20260730c",
            "Standard-fare observation count and share by airline",
            "Vietnam Airlines and VietJet contribute most observations; "
            "smaller carriers remain visible and require support-aware audit.",
            [("5", "airlines"), ("923,855", "total observations")],
        ),
        figure_panel(
            "05",
            "Fare behaviour across booking windows",
            "Median fare and interquartile range summarise price structure at "
            "the eleven observed days-until-departure windows.",
            "/data_analysis/booking_window_fares.png?v=20260730c",
            "Median and interquartile fare across eleven booking windows",
            "Price levels remain relatively stable at long horizons, then "
            "rise more clearly during the final three days before departure.",
            [("11", "canonical windows"), ("DUD 1", "highest median fare")],
        ),
        figure_panel(
            "06",
            "Study-wide temporal fare movement",
            "Observed median fare traces reveal temporal movement and source-"
            "composition change across development and both prospective tests.",
            "/data_analysis/airline_composition_temporal_drift.png?v=20260825a",
            "Observed airline composition and temporal fare drift",
            "The figure reports observed drift directly. Source transition "
            "remains explicit and is not presented as a causal price effect.",
            [("5", "airlines"), ("2", "prospective blocks")],
        ),
        rx.box(
            rx.hstack(
                rx.icon("arrow-right", size=18, color=BLUE),
                rx.vstack(
                    rx.text(
                        "NEXT: FEATURE ENGINEERING",
                        size="1",
                        color=BLUE,
                        weight="bold",
                    ),
                    rx.text(
                        "Raw observations end here. WARM/COLD regimes, "
                        "strictly-prior anchors, candidate provenance, and "
                        "DUD support are derived in the next stage.",
                        size="2",
                        color=INK,
                    ),
                    spacing="0",
                    align="start",
                ),
                spacing="3",
                align="center",
            ),
            background=BLUE_SOFT,
            border_left=f"4px solid {BLUE}",
            padding="1rem 1.25rem",
            width="100%",
        ),
        spacing="5",
        width="100%",
    )


def feature_flow_step(
    icon: str,
    eyebrow: str,
    title: str,
    detail: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.icon(icon, size=19, color=accent),
            width="38px",
            height="38px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=tint,
            border_radius="7px",
        ),
        rx.text(
            eyebrow,
            size="1",
            color=accent,
            weight="bold",
        ),
        rx.text(
            title,
            size="2",
            color=INK,
            weight="bold",
        ),
        rx.text(
            detail,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        spacing="2",
        align="start",
        min_height="176px",
        padding="1.1rem",
        border_left=f"3px solid {accent}",
        background=PANEL,
        width="100%",
    )


def feature_token(
    name: str,
    explanation: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.tooltip(
        rx.box(
            rx.text(
                name,
                size="1",
                color=accent,
                weight="medium",
                font_family=MONO,
                white_space="nowrap",
            ),
            background=tint,
            border=f"1px solid {accent}26",
            border_radius="5px",
            padding="0.34rem 0.52rem",
            cursor="default",
            user_select="text",
        ),
        content=explanation,
        side="top",
        delay_duration=250,
    )


def feature_family_row(
    icon: str,
    title: str,
    purpose: str,
    features: list[tuple[str, str]],
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.icon(icon, size=17, color=accent),
                rx.text(title.upper(), size="1", color=INK, weight="bold"),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.text(
                f"{len(features)} FIELDS",
                size="1",
                color=accent,
                weight="bold",
                font_family=MONO,
            ),
            width="100%",
            align="center",
        ),
        rx.text(
            purpose,
            size="1",
            color=MUTED,
            line_height="1.4",
        ),
        rx.flex(
            *[
                feature_token(name, explanation, accent, tint)
                for name, explanation in features
            ],
            wrap="wrap",
            gap="0.42rem",
            width="100%",
        ),
        spacing="2",
        align="start",
        padding_y="0.85rem",
        border_bottom=f"1px solid {LINE}",
        width="100%",
    )


def task_feature_map(
    task_label: str,
    title: str,
    target: str,
    feature_count: str,
    accent: str,
    tint: str,
    families: list[tuple[str, str, str, list[tuple[str, str]]]],
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text(
                        task_label,
                        size="1",
                        color=accent,
                        weight="bold",
                    ),
                    rx.heading(
                        title,
                        size="5",
                        color=INK,
                        letter_spacing="0",
                    ),
                    spacing="0",
                    align="start",
                ),
                rx.spacer(),
                rx.vstack(
                    rx.text(
                        feature_count,
                        size="5",
                        color=accent,
                        weight="bold",
                        font_family=MONO,
                    ),
                    rx.text(
                        "model-ready fields",
                        size="1",
                        color=MUTED,
                    ),
                    spacing="0",
                    align="end",
                ),
                width="100%",
                align="start",
            ),
            rx.box(
                rx.text(
                    "LEARNING TARGET",
                    size="1",
                    color=accent,
                    weight="bold",
                ),
                rx.text(
                    target,
                    size="1",
                    color=INK,
                    line_height="1.5",
                    font_family=MONO,
                ),
                background=tint,
                border_left=f"3px solid {accent}",
                padding="0.8rem 0.9rem",
                width="100%",
            ),
            rx.vstack(
                *[
                    feature_family_row(
                        icon,
                        family,
                        purpose,
                        features,
                        accent,
                        tint,
                    )
                    for icon, family, purpose, features in families
                ],
                spacing="0",
                align="start",
                width="100%",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="1.25rem",
        width="100%",
    )


def audit_token(name: str, explanation: str = "") -> rx.Component:
    chip = rx.box(
        rx.text(
            name,
            size="1",
            color=BLUE,
            weight="medium",
            font_family=MONO,
            white_space="nowrap",
        ),
        background="#EEF2F7",
        border=f"1px solid {BLUE}24",
        border_radius="5px",
        padding="0.32rem 0.5rem",
        cursor="default",
        user_select="text",
    )
    if not explanation:
        return chip
    return rx.tooltip(
        chip,
        content=explanation,
        side="top",
        delay_duration=250,
    )


def audit_registry_group(
    title: str,
    purpose: str,
    tags: list[tuple[str, str]],
) -> rx.Component:
    return rx.grid(
        rx.vstack(
            rx.text(title, size="1", color=INK, weight="bold"),
            rx.text(
                purpose,
                size="1",
                color=MUTED,
                line_height="1.4",
            ),
            spacing="1",
            align="start",
        ),
        rx.flex(
            *[audit_token(name, explanation) for name, explanation in tags],
            wrap="wrap",
            gap="0.42rem",
            width="100%",
        ),
        grid_template_columns="minmax(190px,0.55fr) minmax(0,1.8fr)",
        gap="1.25rem",
        align_items="start",
        padding="0.9rem 0",
        border_bottom=f"1px solid {LINE}",
        width="100%",
    )


def support_state(
    code: str,
    title: str,
    count: str,
    detail: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.badge(
                code,
                variant="soft",
                color_scheme="teal" if accent == TEAL else "blue",
            ),
            rx.spacer(),
            rx.text(
                count,
                size="1",
                color=accent,
                weight="bold",
                font_family=MONO,
            ),
            width="100%",
            align="center",
        ),
        rx.text(title, size="3", color=INK, weight="bold"),
        rx.text(
            detail,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        spacing="2",
        align="start",
        min_height="142px",
        padding="1rem",
        background=tint,
        border_top=f"3px solid {accent}",
        width="100%",
    )


def governance_row(
    label: str,
    examples: str,
    purpose: str,
    access: str,
    accent: str,
) -> rx.Component:
    return rx.grid(
        rx.hstack(
            rx.box(
                width="8px",
                height="8px",
                border_radius="50%",
                background=accent,
                flex_shrink="0",
            ),
            rx.text(label, size="2", color=INK, weight="bold"),
            spacing="2",
            align="center",
        ),
        rx.text(examples, size="1", color=MUTED, line_height="1.45"),
        rx.text(purpose, size="1", color=INK, line_height="1.45"),
        rx.text(
            access,
            size="1",
            color=accent,
            weight="bold",
            text_align="right",
        ),
        grid_template_columns=(
            "minmax(150px,0.8fr) minmax(260px,1.5fr) "
            "minmax(240px,1.25fr) minmax(110px,0.55fr)"
        ),
        gap="1.25rem",
        align_items="center",
        padding="0.9rem 1rem",
        border_bottom=f"1px solid {LINE}",
        width="100%",
    )


def _legacy_feature_engineering_page() -> rx.Component:
    regression_families = [
        (
            "route",
            "Market identity and session context",
            "Locates each observation within its market and booking session.",
            [
                ("route", "Directed origin-destination market."),
                ("airline", "Operating airline recorded for the schedule slot."),
                ("model_session_label", "Training session label under the AM/PM collection convention."),
                ("departure_period", "Categorical departure-time period."),
            ],
        ),
        (
            "calendar-range",
            "Horizon and calendar",
            "Controls when the requested flight occurs.",
            [
                ("query_dud", "Days between query session and flight date."),
                ("flight_day_of_week", "Calendar weekday of flight departure."),
                ("flight_month", "Calendar month of flight departure."),
                ("is_peak_period", "Indicator for a defined peak travel period."),
                ("departure_time_sin", "Sine encoding of departure minute on a 24-hour cycle."),
                ("departure_time_cos", "Cosine encoding of departure minute on a 24-hour cycle."),
            ],
        ),
        (
            "anchor",
            "Strictly-prior anchor",
            "Provides a legal reference price from a completed prior session.",
            [
                ("prior_anchor_source", "Source category used to construct the strictly-prior anchor."),
                ("prior_anchor_log", "Natural logarithm of the strictly-prior anchor price."),
                ("prior_anchor_support_log1p", "Log-transformed count of observations supporting the anchor."),
                ("prior_anchor_age_hours", "Hours between anchor evidence and query-session label time."),
            ],
        ),
        (
            "chart-no-axes-combined",
            "Prior market context",
            "Summarises market conditions available before the query session.",
            [
                ("prior_market_change_pct_per_day", "Previous market price change normalised per day."),
                ("has_prior_market_change", "Presence indicator for prior market movement."),
                ("prior_competitor_airline_count", "Number of competing airlines in prior evidence."),
                ("prior_competitor_offer_count", "Number of competing offers in prior evidence."),
                ("prior_route_min_log_price", "Log of the prior route-level minimum price."),
                ("prior_route_price_spread_log1p", "Log-transformed prior route price spread."),
            ],
        ),
        (
            "history",
            "Same-slot legal history",
            "Captures legal history for the same schedule-slot proxy.",
            [
                ("history_support_count", "Count of legal prior observations for the same schedule slot."),
                ("is_first_observation", "Marks rows with no legal same-slot history."),
                ("previous_relative_log", "Previous same-slot log price relative to its anchor."),
                ("relative_lag_age_hours", "Age in hours of the previous same-slot relative price."),
                ("prior_relative_volatility", "Historical variation of same-slot relative prices."),
                ("prior_relative_trend_per_dud_day", "Historical same-slot relative trend per DUD day."),
                ("has_previous_same_schedule", "Presence indicator for a previous same-slot observation."),
                ("has_prior_relative_volatility", "Presence indicator for defined same-slot volatility."),
                ("has_prior_relative_trend", "Presence indicator for defined same-slot trend."),
            ],
        ),
        (
            "layers-3",
            "Schedule-template legal history",
            "Backs sparse slots with prior evidence from their schedule template.",
            [
                ("template_history_support_count", "Count of legal prior observations for the schedule template."),
                ("template_previous_relative_log", "Previous template-level log price relative to its anchor."),
                ("template_lag_age_hours", "Age in hours of previous schedule-template evidence."),
                ("template_prior_relative_volatility", "Historical variation of template-relative prices."),
                ("template_prior_relative_trend_per_dud_day", "Template-relative trend per DUD day."),
                ("has_previous_schedule_template", "Presence indicator for previous template evidence."),
                ("has_template_relative_volatility", "Presence indicator for defined template volatility."),
                ("has_template_relative_trend", "Presence indicator for defined template trend."),
            ],
        ),
    ]
    classification_families = [
        (
            "route",
            "Market identity and session context",
            "Locates each transition within its market and booking session.",
            [
                ("route", "Directed origin-destination market."),
                ("airline", "Operating airline recorded for the schedule slot."),
                ("session_label", "Source booking-session label."),
                ("departure_period", "Categorical departure-time period."),
                ("transition", "Current-to-next canonical booking-window transition."),
            ],
        ),
        (
            "calendar-range",
            "Horizon and calendar",
            "Defines current and next canonical booking-window timing.",
            [
                ("days_until_departure", "Current canonical days-until-departure value."),
                ("target_dud", "Next canonical days-until-departure value."),
                ("horizon_gap_days", "Number of days between current and target booking windows."),
                ("flight_day_of_week", "Calendar weekday of flight departure."),
                ("flight_month", "Calendar month of flight departure."),
                ("is_peak_period", "Indicator for a defined peak travel period."),
                ("departure_time_sin", "Sine encoding of departure minute on a 24-hour cycle."),
                ("departure_time_cos", "Cosine encoding of departure minute on a 24-hour cycle."),
            ],
        ),
        (
            "badge-dollar-sign",
            "Current price and anchor context",
            "Expresses current price against legal prior reference evidence.",
            [
                ("anchor_source", "Source category used to construct the strictly-prior anchor."),
                ("log_price_vnd", "Natural logarithm of the current observed price."),
                ("current_relative_log", "Current log price relative to its strictly-prior anchor."),
                ("anchor_support_log1p", "Log-transformed count of observations supporting the anchor."),
            ],
        ),
        (
            "chart-no-axes-combined",
            "Competitor and market context",
            "Positions current price within legally available market evidence.",
            [
                ("competitor_airline_count", "Number of competing airlines available at feature time."),
                ("competitor_offer_count", "Number of competing offers available at feature time."),
                ("log_current_over_competitor_min", "Log ratio of current price to competitor minimum."),
                ("log_same_airline_alt_over_current", "Log ratio of same-airline alternative to current price."),
                ("prior_market_change_pct_per_day", "Previous market price change normalised per day."),
                ("has_prior_market_change", "Presence indicator for prior market movement."),
            ],
        ),
        (
            "history",
            "Same-slot legal history",
            "Captures legal prior movement for the same schedule-slot proxy.",
            [
                ("relative_history_eligible", "Marks availability of legal same-slot relative history."),
                ("previous_relative_log", "Previous same-slot log price relative to its anchor."),
                ("market_shift_log", "Prior market-level log shift associated with the row."),
                ("relative_lag_age_hours", "Age in hours of the previous same-slot relative price."),
                ("prior_relative_count", "Count of legal prior same-slot relative prices."),
                ("prior_relative_volatility", "Historical variation of same-slot relative prices."),
                ("prior_relative_trend_per_dud_day", "Historical same-slot relative trend per DUD day."),
            ],
        ),
    ]
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text(
                        "FEATURE ENGINEERING",
                        size="1",
                        color=TEAL,
                        weight="bold",
                    ),
                    rx.heading(
                        "Turning observations into legal predictive evidence",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                        line_height="1.12",
                        max_width="760px",
                    ),
                    rx.text(
                        "SkyFare derives two task-specific feature frames while "
                        "preserving point-in-time availability. Every predictor "
                        "must exist before its target becomes observable.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                        max_width="760px",
                    ),
                    spacing="3",
                    align="start",
                ),
                rx.box(
                    rx.vstack(
                        rx.text(
                            "FEATURE CONTRACT",
                            size="1",
                            color=BLUE,
                            weight="bold",
                        ),
                        rx.hstack(
                            rx.vstack(
                                rx.text(
                                    "37",
                                    size="7",
                                    color=INK,
                                    weight="bold",
                                    font_family=MONO,
                                ),
                                rx.text(
                                    "Regression fields",
                                    size="1",
                                    color=MUTED,
                                ),
                                spacing="0",
                                align="start",
                            ),
                            rx.box(
                                width="1px",
                                height="48px",
                                background=LINE,
                            ),
                            rx.vstack(
                                rx.text(
                                    "30",
                                    size="7",
                                    color=INK,
                                    weight="bold",
                                    font_family=MONO,
                                ),
                                rx.text(
                                    "Classification fields",
                                    size="1",
                                    color=MUTED,
                                ),
                                spacing="0",
                                align="start",
                            ),
                            spacing="5",
                            align="center",
                        ),
                        rx.text(
                            "Identifiers, targets, fold labels, and future "
                            "observations remain outside predictor matrices.",
                            size="1",
                            color=MUTED,
                            line_height="1.45",
                        ),
                        spacing="3",
                        align="start",
                    ),
                    background=BLUE_SOFT,
                    border_left=f"4px solid {BLUE}",
                    padding="1.25rem",
                    min_height="180px",
                    display="flex",
                    align_items="center",
                ),
                grid_template_columns="minmax(0,1.7fr) minmax(300px,0.8fr)",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border_bottom=f"1px solid {LINE}",
            padding="2.25rem 2rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "POINT-IN-TIME-SAFE TRANSFORMATION",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                rx.heading(
                    "Only evidence available before target time may enter",
                    size="5",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.grid(
                    feature_flow_step(
                        "database",
                        "01 \u00b7 OBSERVE",
                        "Standard fare record",
                        "Route, airline, departure context, booking session, DUD, and observed fare.",
                        TEAL,
                        TEAL_SOFT,
                    ),
                    feature_flow_step(
                        "clock-3",
                        "02 \u00b7 CUT OFF",
                        "Enforce temporal legality",
                        "Feature time must precede label time; query-target fields remain unavailable.",
                        BLUE,
                        BLUE_SOFT,
                    ),
                    feature_flow_step(
                        "history",
                        "03 \u00b7 DERIVE",
                        "Build prior evidence",
                        "Create anchors, market context, legal history, support counts, and presence masks.",
                        TEAL,
                        TEAL_SOFT,
                    ),
                    feature_flow_step(
                        "split",
                        "04 \u00b7 BRANCH",
                        "Create task frames",
                        "Regression and Classification receive different targets and feature contracts.",
                        BLUE,
                        BLUE_SOFT,
                    ),
                    columns="4",
                    spacing="3",
                    width="100%",
                    margin_top="0.75rem",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            background="#F5F9F8",
            padding="1.5rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "TASK-SPECIFIC FEATURE MAPS",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                rx.heading(
                    "One observation base, two learning problems",
                    size="5",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.text(
                    "Shared market context is transformed differently because "
                    "Regression estimates a price level while Classification "
                    "estimates a next-window event probability.",
                    size="2",
                    color=MUTED,
                    line_height="1.55",
                    max_width="850px",
                ),
                rx.grid(
                    task_feature_map(
                        "REGRESSION FRAME",
                        "Regression",
                        "log(query_session_observed_fare_vnd / prior_anchor_vnd)",
                        "5 + 32",
                        TEAL,
                        TEAL_SOFT,
                        regression_families,
                    ),
                    task_feature_map(
                        "CLASSIFICATION FRAME",
                        "Classification",
                        "1 when target_price_vnd <= 0.95 \u00d7 source_price_vnd",
                        "6 + 24",
                        BLUE,
                        BLUE_SOFT,
                        classification_families,
                    ),
                    columns="2",
                    spacing="4",
                    align_items="start",
                    width="100%",
                    margin_top="0.75rem",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.text(
                            "FEATURE GOVERNANCE",
                            size="1",
                            color=BLUE,
                            weight="bold",
                        ),
                        rx.heading(
                            "Audit-only Tag Registry",
                            size="6",
                            color=INK,
                            letter_spacing="0",
                        ),
                        rx.text(
                            "Tags preserve routing, provenance, evaluation, and reporting context.",
                            size="2",
                            color=MUTED,
                            line_height="1.5",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    rx.spacer(),
                    rx.badge(
                        "MODEL ACCESS: BLOCKED",
                        color_scheme="red",
                        variant="soft",
                        size="2",
                    ),
                    width="100%",
                    align="center",
                ),
                audit_registry_group(
                    "IDENTITY AND CANDIDATE",
                    "Pairs rows and proves candidate provenance.",
                    [
                        ("query_id", "Groups ranked schedule-slot predictions for one query."),
                        ("schedule_slot_id", "Proxy identity used for pairing, grouping, and audit only."),
                        ("schedule_template_id", "Groups route-airline-departure templates for audit."),
                        ("identity_quality", "Quality state of the schedule-slot identity match."),
                        ("is_schedule_fallback", "Marks candidate schedules produced through fallback."),
                        ("candidate_source", "Prior cache or database source that supplied the candidate."),
                        ("route_airline", "Route-airline audit grouping key."),
                    ],
                ),
                audit_registry_group(
                    "MARKET GROUP",
                    "Reports behaviour across market structures.",
                    [
                        ("TRUNK", "High-volume domestic trunk routes."),
                        ("TOURISM", "Routes serving major tourism markets."),
                        ("REGIONAL_ALTERNATIVE", "Regional or alternative-airport routes."),
                    ],
                ),
                audit_registry_group(
                    "REGIME AND HISTORY SUPPORT",
                    "Describes legal evidence available for each row.",
                    [
                        ("regime", "Information regime assigned from prior support."),
                        ("WARM", "Row has established legal history."),
                        ("COLD", "Row has insufficient legal history for warm routing."),
                        ("history_support_band", "Fold-local history support category."),
                        ("template_history_support_band", "Fold-local schedule-template history support category."),
                        ("FIRST_0", "No prior same-slot observations."),
                        ("COLD_1_2", "One or two prior same-slot observations."),
                        ("WARM_3_PLUS", "At least three prior same-slot observations."),
                    ],
                ),
                audit_registry_group(
                    "COVERAGE AND SUPPORT",
                    "Reports airline breadth and fold-local training support.",
                    [
                        ("coverage_band", "Number of airlines observed for the route."),
                        ("UNIVERSAL_5", "All five airlines represented."),
                        ("BROAD_4_5", "Four or five airlines represented."),
                        ("MID_2_3", "Two or three airlines represented."),
                        ("SINGLE_1", "One airline represented."),
                        ("UNOBSERVED_0", "No airline coverage observed in the legal training split."),
                        ("support_tier", "High-level route support grouping."),
                        ("HIGH_SUPPORT", "Broad route coverage."),
                        ("LOWER_SUPPORT", "Narrower route coverage."),
                        ("route_support_quartile", "Fold-local route support quartile."),
                        ("Q1_LOW", "Lowest route-support quartile."),
                        ("Q2", "Second route-support quartile."),
                        ("Q3", "Third route-support quartile."),
                        ("Q4_HIGH", "Highest route-support quartile."),
                        ("UNSEEN", "Route absent from the legal training split."),
                        ("train_route_airline_support", "Training-row count for the exact route-airline pair."),
                        ("route_airline_support_band", "Band derived from exact route-airline training support."),
                        ("UNSEEN_0", "No route-airline training observations."),
                        ("LOW_1_99", "One to ninety-nine supporting observations."),
                        ("MEDIUM_100_499", "One hundred to four hundred ninety-nine observations."),
                        ("HIGH_500_1999", "Five hundred to one thousand nine hundred ninety-nine observations."),
                        ("VERY_HIGH_2000_PLUS", "At least two thousand supporting observations."),
                    ],
                ),
                audit_registry_group(
                    "ANCHOR PROVENANCE (NON-PREDICTOR)",
                    "Reports strength, age, collection period, and fallback state.",
                    [
                        ("anchor_fallback_level", "Hierarchy level that supplied the anchor."),
                        ("OBSERVED_PEER", "Classification anchor came from an observed peer group."),
                        ("FALLBACK", "Classification anchor used a fallback source."),
                        ("ROUTE_AIRLINE_PERIOD_DUD", "Most specific route-airline-period-DUD anchor."),
                        ("ROUTE_AIRLINE_DUD", "Route-airline-DUD fallback anchor."),
                        ("ROUTE_DUD", "Route-DUD fallback anchor."),
                        ("AIRLINE_DUD", "Airline-DUD fallback anchor."),
                        ("GLOBAL_DUD", "Global DUD fallback anchor."),
                        ("GLOBAL_BATCH", "Completed prior-batch median fallback."),
                        ("UNAVAILABLE", "No legal anchor was available."),
                        ("anchor_support_band", "Band derived from anchor support count."),
                        ("NONE_0", "No classification anchor support observations."),
                        ("LOW_1_9", "One to nine classification anchor observations."),
                        ("MEDIUM_10_49", "Ten to forty-nine classification anchor observations."),
                        ("HIGH_50_PLUS", "At least fifty classification anchor observations."),
                        ("UNSEEN_0", "No anchor support observations."),
                        ("LOW_1_99", "One to ninety-nine anchor observations."),
                        ("MEDIUM_100_499", "One hundred to four hundred ninety-nine anchor observations."),
                        ("HIGH_500_1999", "Five hundred to one thousand nine hundred ninety-nine anchor observations."),
                        ("VERY_HIGH_2000_PLUS", "At least two thousand anchor observations."),
                        ("anchor_age_band", "Band derived from anchor evidence age."),
                        ("FRESH_LE_12H", "Anchor is no older than twelve hours."),
                        ("AGE_12_24H", "Anchor age is between twelve and twenty-four hours."),
                        ("AGE_24_48H", "Anchor age is between twenty-four and forty-eight hours."),
                        ("STALE_GT_48H", "Anchor is older than forty-eight hours."),
                        ("anchor_collection_era", "Collection era supplying anchor evidence."),
                        ("anchor_is_fallback", "Marks anchors below the most specific hierarchy level."),
                    ],
                ),
                audit_registry_group(
                    "DUD SUPPORT",
                    "Separates direct model support from interpolation and extrapolation.",
                    [
                        ("dud_support_mode", "Support mode for requested days until departure."),
                        ("ON_GRID", "Requested DUD is a canonical booking window."),
                        ("INTERIOR_OFF_GRID_INTERPOLATED", "Requested DUD is interpolated between canonical windows."),
                        ("ENDPOINT_EXTRAPOLATED", "Requested DUD lies beyond an observed endpoint."),
                        ("OUT_OF_SCOPE", "Requested DUD is outside supported scope."),
                    ],
                ),
                audit_registry_group(
                    "TARGET OBSERVABILITY",
                    "Controls evaluation eligibility without becoming a predictor.",
                    [
                        ("target_batch_exists", "Whether the required target collection batch exists."),
                        ("target_observation_state", "Maturity and observation state of the target."),
                        ("OBSERVED", "Mature target price was observed."),
                        ("MATURE_NOT_OBSERVED", "Target matured but matching slot was not observed."),
                        ("IMMATURE", "Target time had not occurred by cutoff."),
                        ("TARGET_BATCH_NOT_COLLECTED", "Target time occurred but required batch was not collected."),
                    ],
                ),
                audit_registry_group(
                    "TEMPORAL AND PREDICTION AUDIT",
                    "Reconstructs when and how each prediction was produced.",
                    [
                        ("feature_time", "Latest timestamp legally available to feature construction."),
                        ("source_session_key", "Source session identifier for paired transition audit."),
                        ("target_session_key", "Target session identifier for paired evaluation."),
                        ("label_time", "Timestamp at which target becomes observable."),
                        ("collection_era", "Collection period associated with the observation."),
                        ("target_collection_era", "Collection period associated with the target."),
                        ("source_target_era_transition", "Source-to-target collection-period transition."),
                        ("bridge_label_stability", "Audit state for labels spanning collection periods."),
                        ("fold", "Temporal split identifier."),
                        ("fold_role", "Purpose assigned to the temporal split."),
                        ("data_cutoff", "Latest allowed timestamp for the artifact."),
                        ("prediction_path", "Runtime path used to produce the prediction."),
                        ("hierarchy_level", "Fallback hierarchy level used at prediction time."),
                        ("model_version", "Version of the frozen model artifact."),
                        ("feature_contract_version", "Version of the frozen feature contract."),
                        ("baseline_version", "Version of the comparison baseline."),
                    ],
                ),
                rx.box(
                    rx.grid(
                        rx.text("FIELD CLASS", size="1", color=MUTED, weight="bold"),
                        rx.text("PURPOSE", size="1", color=MUTED, weight="bold"),
                        rx.text(
                            "MODEL ACCESS",
                            size="1",
                            color=MUTED,
                            weight="bold",
                            text_align="right",
                        ),
                        grid_template_columns=(
                            "minmax(180px,0.7fr) minmax(400px,1.8fr) "
                            "minmax(120px,0.5fr)"
                        ),
                        gap="1.25rem",
                        padding="0.8rem 1rem",
                        background="#F5F7F8",
                        width="100%",
                    ),
                    rx.grid(
                        rx.text("Predictors", size="2", color=INK, weight="bold"),
                        rx.text(
                            "Point-in-time-safe fields listed in Regression and Classification inventories.",
                            size="1",
                            color=MUTED,
                        ),
                        rx.text("ALLOWED", size="1", color=GREEN, weight="bold", text_align="right"),
                        grid_template_columns="minmax(180px,0.7fr) minmax(400px,1.8fr) minmax(120px,0.5fr)",
                        gap="1.25rem",
                        padding="0.85rem 1rem",
                        border_bottom=f"1px solid {LINE}",
                        width="100%",
                    ),
                    rx.grid(
                        rx.text("Audit-only tags", size="2", color=INK, weight="bold"),
                        rx.text(
                            "Identity, routing, support, provenance, observability, and prediction audit.",
                            size="1",
                            color=MUTED,
                        ),
                        rx.text("BLOCKED", size="1", color=BLUE, weight="bold", text_align="right"),
                        grid_template_columns="minmax(180px,0.7fr) minmax(400px,1.8fr) minmax(120px,0.5fr)",
                        gap="1.25rem",
                        padding="0.85rem 1rem",
                        border_bottom=f"1px solid {LINE}",
                        width="100%",
                    ),
                    rx.grid(
                        rx.text("Target-only fields", size="2", color=INK, weight="bold"),
                        rx.text(
                            "Observed prices and supervised labels used only after target maturity.",
                            size="1",
                            color=MUTED,
                        ),
                        rx.text("BLOCKED", size="1", color=CORAL, weight="bold", text_align="right"),
                        grid_template_columns="minmax(180px,0.7fr) minmax(400px,1.8fr) minmax(120px,0.5fr)",
                        gap="1.25rem",
                        padding="0.85rem 1rem",
                        border_bottom=f"1px solid {LINE}",
                        width="100%",
                    ),
                    rx.grid(
                        rx.text("Excluded identifiers", size="2", color=INK, weight="bold"),
                        rx.text(
                            "Raw identifiers, timestamps, fold labels, and output-path fields excluded from model matrices.",
                            size="1",
                            color=MUTED,
                        ),
                        rx.text("BLOCKED", size="1", color=AMBER, weight="bold", text_align="right"),
                        grid_template_columns="minmax(180px,0.7fr) minmax(400px,1.8fr) minmax(120px,0.5fr)",
                        gap="1.25rem",
                        padding="0.85rem 1rem",
                        width="100%",
                    ),
                    border=f"1px solid {LINE}",
                    border_radius="8px",
                    overflow="hidden",
                    width="100%",
                    margin_top="0.75rem",
                ),
                rx.hstack(
                    rx.icon("arrow-right", size=18, color=BLUE),
                    rx.text(
                        "Post-fit grouped importance and ablation results are reported under Evaluation.",
                        size="1",
                        color=INK,
                        weight="bold",
                    ),
                    spacing="2",
                    align="center",
                    background=BLUE_SOFT,
                    padding="0.85rem 1rem",
                    width="100%",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            background=PANEL,
            border_top=f"4px solid {BLUE}",
            padding="1.5rem",
            width="100%",
        ),
        spacing="6",
        width="100%",
    )


def _feature_family_cell(
    spec: tuple[str, str, str, list[tuple[str, str]]],
    accent: str,
    tint: str,
) -> rx.Component:
    icon, title, purpose, features = spec
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.icon(icon, size=17, color=accent),
                rx.text(title.upper(), size="1", color=INK, weight="bold"),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.text(
                f"{len(features)} FIELDS",
                size="1",
                color=accent,
                weight="bold",
                font_family=MONO,
                white_space="nowrap",
            ),
            width="100%",
            align="center",
        ),
        rx.text(purpose, size="1", color=MUTED, line_height="1.45"),
        rx.flex(
            *[
                feature_token(name, explanation, accent, tint)
                for name, explanation in features
            ],
            wrap="wrap",
            gap="0.42rem",
            width="100%",
        ),
        spacing="2",
        align="start",
        padding="1rem 1.1rem",
        min_height="100%",
        width="100%",
    )


def _task_map_header(
    eyebrow: str,
    title: str,
    target: str,
    count: str,
    count_label: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.text(eyebrow, size="1", color=accent, weight="bold"),
                rx.heading(title, size="5", color=INK, letter_spacing="0"),
                spacing="0",
                align="start",
            ),
            rx.spacer(),
            rx.vstack(
                rx.text(
                    count,
                    size="5",
                    color=accent,
                    weight="bold",
                    font_family=MONO,
                ),
                rx.text(count_label, size="1", color=MUTED),
                spacing="0",
                align="end",
            ),
            width="100%",
            align="start",
        ),
        rx.box(
            rx.text("LEARNING TARGET", size="1", color=accent, weight="bold"),
            rx.text(
                target,
                size="1",
                color=INK,
                line_height="1.5",
                font_family=MONO,
            ),
            background=tint,
            border_left=f"3px solid {accent}",
            padding="0.8rem 0.9rem",
            min_height="104px",
            width="100%",
        ),
        spacing="3",
        align="start",
        padding="1.2rem",
        min_height="196px",
        width="100%",
    )


def _governance_lane(
    title: str,
    detail: str,
    access: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(title, size="2", color=INK, weight="bold"),
            rx.spacer(),
            rx.box(
                rx.text(access, size="1", color=accent, weight="bold"),
                background=tint,
                border_radius="5px",
                padding="0.24rem 0.45rem",
            ),
            width="100%",
            align="center",
        ),
        rx.text(detail, size="1", color=MUTED, line_height="1.45"),
        spacing="2",
        align="start",
        padding="0.95rem",
        border_top=f"3px solid {accent}",
        background=PANEL,
        width="100%",
    )


def _audit_token(value: str) -> rx.Component:
    return rx.box(
        rx.text(
            value,
            size="1",
            color="#5E5548",
            font_family=MONO,
            line_height="1.35",
        ),
        background="#F3EFE7",
        border="1px solid #E3D9C9",
        border_radius="5px",
        padding="0.23rem 0.38rem",
        cursor="default",
        width="fit-content",
    )


def _audit_table_cell(
    groups: list[tuple[str, list[str]]],
) -> rx.Component:
    if not groups:
        return rx.text("\u2014", size="2", color="#9B9286")
    return rx.vstack(
        *[
            rx.vstack(
                _audit_token(field),
                rx.text(
                    " \u00b7 ".join(values),
                    size="1",
                    color="#8A7F70",
                    font_family=MONO,
                    line_height="1.45",
                )
                if values
                else rx.fragment(),
                spacing="1",
                align="start",
                width="100%",
            )
            for field, values in groups
        ],
        spacing="2",
        align="start",
        width="100%",
    )


def _audit_registry_row(
    family: str,
    regression: list[tuple[str, list[str]]],
    classification: list[tuple[str, list[str]]],
    common: list[tuple[str, list[str]]],
) -> rx.Component:
    cell_style = {
        "vertical_align": "top",
        "padding": "0.75rem 0.8rem",
        "border_bottom": "1px solid #E3D9C9",
    }
    return rx.table.row(
        rx.table.row_header_cell(
            rx.text(family, size="1", color=INK, weight="bold"),
            **cell_style,
        ),
        rx.table.cell(_audit_table_cell(regression), **cell_style),
        rx.table.cell(_audit_table_cell(classification), **cell_style),
        rx.table.cell(_audit_table_cell(common), **cell_style),
    )


def _audit_only_registry() -> rx.Component:
    rows = [
        (
            "Identity and candidate",
            [("query_id", []), ("schedule_template_id", [])],
            [],
            [
                ("schedule_slot_id", []),
                ("identity_quality", []),
                ("is_schedule_fallback", []),
                ("candidate_source", []),
                ("route_airline", []),
            ],
        ),
        (
            "Market group",
            [],
            [],
            [("market_group", ["TRUNK", "TOURISM", "REGIONAL_ALTERNATIVE"])],
        ),
        (
            "Regime and history support",
            [
                ("template_history_support_count", []),
                ("template_history_support_band", []),
            ],
            [],
            [
                ("regime", ["WARM", "COLD"]),
                ("history_support_count", []),
                ("history_support_band", ["FIRST_0", "COLD_1_2", "WARM_3_PLUS"]),
            ],
        ),
        (
            "Coverage and support",
            [],
            [],
            [
                ("coverage_band", []),
                ("support_tier", []),
                ("route_support_quartile", []),
                ("train_route_airline_support", []),
                ("route_airline_support_band", []),
            ],
        ),
        (
            "Anchor provenance",
            [("prior_anchor_source", [])],
            [("anchor_source", [])],
            [
                ("anchor_fallback_level", []),
                ("anchor_support_band", []),
                ("anchor_age_band", []),
                ("anchor_collection_era", []),
                ("anchor_is_fallback", []),
            ],
        ),
        (
            "DUD support",
            [],
            [],
            [
                (
                    "dud_support_mode",
                    [
                        "ON_GRID",
                        "INTERIOR_OFF_GRID_INTERPOLATED",
                        "ENDPOINT_EXTRAPOLATED",
                        "OUT_OF_SCOPE",
                    ],
                )
            ],
        ),
        (
            "Target observability",
            [],
            [],
            [
                ("target_batch_exists", []),
                (
                    "target_observation_state",
                    [
                        "OBSERVED",
                        "MATURE_NOT_OBSERVED",
                        "IMMATURE",
                        "TARGET_BATCH_NOT_COLLECTED",
                    ],
                ),
            ],
        ),
        (
            "Temporal and prediction audit",
            [],
            [
                ("source_session_key", []),
                ("target_session_key", []),
                ("target_collection_era", []),
                ("source_target_era_transition", []),
                ("bridge_label_stability", []),
            ],
            [
                ("feature_time", []),
                ("label_time", []),
                ("collection_era", []),
                ("fold", []),
                ("fold_role", []),
                ("data_cutoff", []),
                ("prediction_path", []),
                ("hierarchy_level", []),
                ("model_version", []),
                ("feature_contract_version", []),
                ("baseline_version", []),
            ],
        ),
    ]
    header_style = {
        "background": "#F1EADF",
        "color": INK,
        "font_weight": "700",
        "padding": "0.72rem 0.8rem",
        "border_bottom": "1px solid #D8CCBB",
        "white_space": "nowrap",
    }
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text("AUDIT-ONLY TAG REGISTRY", size="1", color=AMBER, weight="bold"),
                    rx.heading(
                        "Stored context, blocked model access",
                        size="5",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        "Routing, provenance, and evaluation context remain stored "
                        "beside observations but outside predictor matrices.",
                        size="2",
                        color=MUTED,
                        line_height="1.5",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                rx.badge(
                    "MODEL ACCESS: BLOCKED",
                    color_scheme="red",
                    variant="soft",
                    size="2",
                ),
                width="100%",
                align="center",
            ),
            rx.box(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Tag family", **header_style),
                            rx.table.column_header_cell("Regression", **header_style),
                            rx.table.column_header_cell("Classification", **header_style),
                            rx.table.column_header_cell("Common", **header_style),
                        )
                    ),
                    rx.table.body(
                        *[
                            _audit_registry_row(
                                family,
                                regression,
                                classification,
                                common,
                            )
                            for family, regression, classification, common in rows
                        ]
                    ),
                    variant="surface",
                    size="2",
                    width="100%",
                    min_width="1180px",
                    background="#FBF7EE",
                ),
                border="1px solid #DED4C3",
                border_radius="8px",
                overflow_x="auto",
                width="100%",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        background="#FBF7EE",
        border_top=f"4px solid {AMBER}",
        padding="1.35rem",
        width="100%",
    )


def feature_engineering_page() -> rx.Component:
    fare_families = [
        (
            "route",
            "Market identity and session context",
            "Locates each observation within its market and booking session.",
            [
                ("route", "Directed origin-destination market."),
                ("airline", "Operating airline recorded for the schedule slot."),
                ("model_session_label", "Training-session label under AM/PM collection convention."),
                ("departure_period", "Categorical departure-time period."),
            ],
        ),
        (
            "calendar-range",
            "Horizon and calendar",
            "Controls when the requested flight occurs.",
            [
                ("query_dud", "Days between query session and flight date."),
                ("flight_day_of_week", "Calendar weekday of flight departure."),
                ("flight_month", "Calendar month of flight departure."),
                ("is_peak_period", "Indicator for a defined peak travel period."),
                ("departure_time_sin", "Sine encoding of departure minute on a 24-hour cycle."),
                ("departure_time_cos", "Cosine encoding of departure minute on a 24-hour cycle."),
            ],
        ),
        (
            "anchor",
            "Strictly-prior anchor",
            "Provides a legal reference price from a completed prior session.",
            [
                ("prior_anchor_source", "Categorical source for strictly-prior anchor."),
                ("prior_anchor_log", "Natural logarithm of strictly-prior anchor price."),
                ("prior_anchor_support_log1p", "Log-transformed count supporting the anchor."),
                ("prior_anchor_age_hours", "Hours between anchor evidence and query feature time."),
            ],
        ),
        (
            "chart-no-axes-combined",
            "Prior market context",
            "Summarises market conditions available before query time.",
            [
                ("prior_market_change_pct_per_day", "Previous market price change per day."),
                ("has_prior_market_change", "Presence indicator for prior market movement."),
                ("prior_competitor_airline_count", "Competing airlines in prior evidence."),
                ("prior_competitor_offer_count", "Competing offers in prior evidence."),
                ("prior_route_min_log_price", "Log of prior route-level minimum price."),
                ("prior_route_price_spread_log1p", "Log-transformed prior route price spread."),
            ],
        ),
        (
            "history",
            "Same-slot legal history",
            "Captures legal history for the same schedule-slot proxy.",
            [
                ("history_support_count", "Legal prior observations for same schedule slot."),
                ("is_first_observation", "Marks no legal same-slot history."),
                ("previous_relative_log", "Previous same-slot log price relative to anchor."),
                ("relative_lag_age_hours", "Age of previous same-slot relative price."),
                ("prior_relative_volatility", "Variation of prior same-slot relative prices."),
                ("prior_relative_trend_per_dud_day", "Prior same-slot trend per DUD day."),
                ("has_previous_same_schedule", "Availability of previous same-slot observation."),
                ("has_prior_relative_volatility", "Availability of same-slot volatility."),
                ("has_prior_relative_trend", "Availability of same-slot trend."),
            ],
        ),
    ]
    exact_families = [
        (
            "route",
            "Market identity and session context",
            "Locates each transition within its market and booking session.",
            [
                ("route", "Directed origin-destination market."),
                ("airline", "Operating airline recorded for the schedule slot."),
                ("session_label", "Source booking-session label."),
                ("departure_period", "Categorical departure-time period."),
                ("transition", "Current-to-next canonical booking-window transition."),
            ],
        ),
        (
            "calendar-range",
            "Horizon and calendar",
            "Defines current and next canonical booking-window timing.",
            [
                ("days_until_departure", "Current canonical DUD."),
                ("target_dud", "Next canonical DUD."),
                ("horizon_gap_days", "Days between current and target windows."),
                ("flight_day_of_week", "Calendar weekday of flight departure."),
                ("flight_month", "Calendar month of flight departure."),
                ("is_peak_period", "Indicator for a defined peak travel period."),
                ("departure_time_sin", "Sine encoding of departure minute."),
                ("departure_time_cos", "Cosine encoding of departure minute."),
            ],
        ),
        (
            "badge-dollar-sign",
            "Current price and anchor context",
            "Expresses current price against legal prior reference evidence.",
            [
                ("anchor_source", "Categorical source used for strictly-prior anchor."),
                ("log_price_vnd", "Natural logarithm of current observed fare."),
                ("current_relative_log", "Current log fare relative to anchor."),
                ("anchor_support_log1p", "Log-transformed count supporting anchor."),
            ],
        ),
        (
            "chart-no-axes-combined",
            "Competitor and market context",
            "Positions current fare within legally available market evidence.",
            [
                ("competitor_airline_count", "Competing airlines available at feature time."),
                ("competitor_offer_count", "Competing offers available at feature time."),
                ("log_current_over_competitor_min", "Log ratio of current fare to competitor minimum."),
                ("log_same_airline_alt_over_current", "Log ratio of same-airline alternative to current fare."),
                ("prior_market_change_pct_per_day", "Previous market price change per day."),
                ("has_prior_market_change", "Presence indicator for prior market movement."),
            ],
        ),
        (
            "history",
            "Same-slot legal history",
            "Captures legal prior movement for the same schedule-slot proxy.",
            [
                ("relative_history_eligible", "Availability gate for legal relative history."),
                ("previous_relative_log", "Previous same-slot log fare relative to anchor."),
                ("market_shift_log", "Prior market-level log shift."),
                ("relative_lag_age_hours", "Age of previous same-slot relative fare."),
                ("prior_relative_count", "Count of legal prior relative fares."),
                ("prior_relative_volatility", "Variation of prior same-slot relative fares."),
                ("prior_relative_trend_per_dud_day", "Prior same-slot trend per DUD day."),
            ],
        ),
    ]
    template_family = (
        "layers-3",
        "Schedule-template legal history",
        "Backs sparse schedule slots with legal prior evidence from their route-airline-departure template.",
        [
            ("template_history_support_count", "Legal prior observations for schedule template."),
            ("template_previous_relative_log", "Previous template log fare relative to anchor."),
            ("template_lag_age_hours", "Age of previous template evidence."),
            ("template_prior_relative_volatility", "Variation of template-relative fares."),
            ("template_prior_relative_trend_per_dud_day", "Template-relative trend per DUD day."),
            ("has_previous_schedule_template", "Availability of prior template evidence."),
            ("has_template_relative_volatility", "Availability of template volatility."),
            ("has_template_relative_trend", "Availability of template trend."),
        ],
    )
    paired_rows = [
        rx.grid(
            _feature_family_cell(fare, TEAL, TEAL_SOFT),
            _feature_family_cell(exact, BLUE, BLUE_SOFT),
            grid_template_columns="repeat(2,minmax(0,1fr))",
            align_items="stretch",
            border_top=f"1px solid {LINE}",
            width="100%",
        )
        for fare, exact in zip(fare_families, exact_families, strict=True)
    ]
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text("FEATURE ENGINEERING", size="1", color=TEAL, weight="bold"),
                    rx.heading(
                        "Turning observations into legal predictive evidence",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                        line_height="1.12",
                    ),
                    rx.text(
                        "Two frozen feature contracts transform the same standard "
                        "observation base without exposing future information.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                    ),
                    spacing="3",
                    align="start",
                ),
                rx.hstack(
                    rx.vstack(
                        rx.text("37", size="7", color=TEAL, weight="bold", font_family=MONO),
                        rx.text("Regression fields", size="1", color=MUTED),
                        spacing="0",
                        align="start",
                    ),
                    rx.box(width="1px", height="48px", background=LINE),
                    rx.vstack(
                        rx.text("30", size="7", color=BLUE, weight="bold", font_family=MONO),
                        rx.text("Classification fields", size="1", color=MUTED),
                        spacing="0",
                        align="start",
                    ),
                    spacing="5",
                    align="center",
                    justify="end",
                ),
                grid_template_columns="minmax(0,1.8fr) minmax(300px,0.7fr)",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border_bottom=f"1px solid {LINE}",
            padding="2.1rem 2rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "01 \u00b7 POINT-IN-TIME-SAFE TRANSFORMATION",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                rx.heading(
                    "Only evidence available before target time may enter",
                    size="5",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.grid(
                    feature_flow_step(
                        "database",
                        "01 \u00b7 OBSERVE",
                        "Standard observation",
                        "Route, airline, departure context, booking session, DUD, and observed fare.",
                        TEAL,
                        TEAL_SOFT,
                    ),
                    feature_flow_step(
                        "clock-3",
                        "02 \u00b7 CUT OFF",
                        "Enforce temporal legality",
                        "Feature time precedes label time; target observations remain unavailable.",
                        BLUE,
                        BLUE_SOFT,
                    ),
                    feature_flow_step(
                        "history",
                        "03 \u00b7 DERIVE",
                        "Build legal prior evidence",
                        "Create anchors, market context, history, support counts, and availability masks.",
                        TEAL,
                        TEAL_SOFT,
                    ),
                    feature_flow_step(
                        "split",
                        "04 \u00b7 BRANCH",
                        "Create task frames",
                        "Regression and Classification receive different targets and frozen contracts.",
                        BLUE,
                        BLUE_SOFT,
                    ),
                    columns="4",
                    spacing="3",
                    width="100%",
                    margin_top="0.7rem",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            background="#F5F9F8",
            padding="1.5rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.text("02 \u00b7 TASK-SPECIFIC FEATURE MAPS", size="1", color=TEAL, weight="bold"),
                rx.heading(
                    "One observation base, two learning problems",
                    size="5",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.text(
                    "Corresponding feature families share rows so differences in "
                    "purpose and field coverage remain directly comparable.",
                    size="2",
                    color=MUTED,
                    line_height="1.55",
                ),
                rx.box(
                    rx.grid(
                        _task_map_header(
                            "REGRESSION",
                            "Regression",
                            "log(query_session_observed_fare_vnd / prior_anchor_vnd)",
                            "5 + 32",
                            "categorical + numeric",
                            TEAL,
                            TEAL_SOFT,
                        ),
                        _task_map_header(
                            "CLASSIFICATION",
                            "Classification",
                            "1 when target_price_vnd <= 0.95 \u00d7 source_price_vnd",
                            "6 + 24",
                            "categorical + numeric",
                            BLUE,
                            BLUE_SOFT,
                        ),
                        grid_template_columns="repeat(2,minmax(0,1fr))",
                        align_items="stretch",
                        width="100%",
                    ),
                    *paired_rows,
                    rx.box(
                        rx.hstack(
                            rx.vstack(
                                rx.text(
                                    "TASK-SPECIFIC EVIDENCE \u00b7 REGRESSION ONLY",
                                    size="1",
                                    color=AMBER,
                                    weight="bold",
                                ),
                                rx.heading(
                                    "Schedule-template legal history",
                                    size="4",
                                    color=INK,
                                    letter_spacing="0",
                                ),
                                spacing="0",
                                align="start",
                            ),
                            rx.spacer(),
                            rx.text(
                                "8 FIELDS",
                                size="1",
                                color=AMBER,
                                weight="bold",
                                font_family=MONO,
                            ),
                            width="100%",
                            align="center",
                        ),
                        rx.text(
                            template_family[2],
                            size="1",
                            color=MUTED,
                            line_height="1.45",
                            margin_top="0.35rem",
                        ),
                        rx.flex(
                            *[
                                feature_token(name, explanation, TEAL, TEAL_SOFT)
                                for name, explanation in template_family[3]
                            ],
                            wrap="wrap",
                            gap="0.42rem",
                            width="100%",
                            margin_top="0.7rem",
                        ),
                        background="#FBF7EE",
                        border_top=f"3px solid {AMBER}",
                        padding="1rem 1.1rem",
                        width="100%",
                    ),
                    border=f"1px solid {LINE}",
                    border_radius="8px",
                    overflow="hidden",
                    width="100%",
                    margin_top="0.65rem",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            width="100%",
        ),
        _audit_only_registry(),
        rx.box(
            rx.image(
                src="/feature_engineering/regression_feature_association.png?v=20260805a",
                alt=(
                    "Lower-triangle mixed-type feature association heatmap for "
                    "all 37 Regression predictors"
                ),
                width="100%",
                height="auto",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            padding="0.75rem",
            width="100%",
            overflow_x="auto",
        ),
        rx.box(
            rx.image(
                src="/feature_engineering/classification_feature_association.png?v=20260805a",
                alt=(
                    "Lower-triangle mixed-type feature association heatmap for "
                    "all 30 Classification predictors"
                ),
                width="100%",
                height="auto",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            padding="0.75rem",
            width="100%",
            overflow_x="auto",
        ),
        rx.text(
            "Grouped permutation importance and ablation remain under Evaluation.",
            size="1",
            color=MUTED,
            font_style="italic",
        ),
        spacing="6",
        width="100%",
    )


def modelling_stat(value: str, label: str, accent: str) -> rx.Component:
    return rx.vstack(
        rx.text(
            value,
            size="6",
            color=accent,
            weight="bold",
            font_family=MONO,
        ),
        rx.text(
            label,
            size="1",
            color=MUTED,
            text_align="center",
            line_height="1.35",
        ),
        spacing="0",
        align="center",
        min_width="112px",
    )


def modelling_section_heading(
    number: str,
    title: str,
    description: str,
) -> rx.Component:
    return rx.vstack(
        rx.text(
            number,
            size="1",
            color=TEAL,
            weight="bold",
            font_family=MONO,
        ),
        rx.heading(
            title,
            size="6",
            color=INK,
            letter_spacing="0",
        ),
        rx.text(
            description,
            size="2",
            color=MUTED,
            line_height="1.55",
            max_width="820px",
        ),
        spacing="1",
        align="start",
        width="100%",
    )


def problem_definition(
    label: str,
    output_type: str,
    target: str,
    reconstruction: str,
    output: str,
    accent: str,
    tint: str,
    icon: str,
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(icon, size=21, color=accent),
                    width="42px",
                    height="42px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    background=tint,
                    border_radius="7px",
                    flex_shrink="0",
                ),
                rx.vstack(
                    rx.text(label, size="4", color=INK, weight="bold"),
                    rx.text(
                        output_type,
                        size="1",
                        color=accent,
                        weight="bold",
                    ),
                    spacing="0",
                    align="start",
                ),
                spacing="3",
                align="center",
            ),
            rx.box(
                rx.text(
                    target,
                    size="2",
                    color=INK,
                    font_family=MONO,
                    line_height="1.55",
                ),
                background="#F7F9F9",
                border_left=f"3px solid {accent}",
                padding="0.8rem 0.95rem",
                width="100%",
            ),
            rx.vstack(
                rx.text(
                    "MODEL OUTPUT",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                rx.text(
                    reconstruction,
                    size="2",
                    color=INK,
                    font_family=MONO,
                    line_height="1.5",
                ),
                rx.text(
                    output,
                    size="1",
                    color=MUTED,
                    line_height="1.5",
                ),
                spacing="1",
                align="start",
                width="100%",
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        background=PANEL,
        border_top=f"3px solid {accent}",
        border_bottom=f"1px solid {LINE}",
        padding="1.25rem",
        min_height="292px",
        width="100%",
    )


def modelling_figure(
    image: str,
    alt: str,
    note: str,
    accent: str,
    max_width: str = "100%",
) -> rx.Component:
    return rx.vstack(
        rx.image(
            src=image,
            alt=alt,
            width="100%",
            max_width=max_width,
            height="auto",
            margin_x="auto",
        ),
        rx.hstack(
            rx.icon("info", size=16, color=accent),
            rx.text(
                note,
                size="1",
                color=MUTED,
                line_height="1.45",
            ),
            spacing="2",
            align="start",
            width="100%",
            padding_top="0.65rem",
            border_top=f"1px solid {LINE}",
        ),
        spacing="2",
        align="start",
        width="100%",
    )


def modelling_temporal_figure(
    number: str,
    title: str,
    description: str,
    image: str,
    alt: str,
    takeaway: str,
    stats: list[tuple[str, str]],
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.grid(
                rx.vstack(
                    rx.text(
                        f"FIGURE {number} \u00b7 TEMPORAL DEVELOPMENT",
                        size="1",
                        color=accent,
                        weight="bold",
                        font_family=MONO,
                    ),
                    rx.heading(
                        title,
                        size="5",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        description,
                        size="2",
                        color=MUTED,
                        line_height="1.55",
                        max_width="720px",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.hstack(
                    *[
                        evidence_stat(value, label, accent)
                        for value, label in stats
                    ],
                    spacing="4",
                    align="center",
                    justify="end",
                    width="100%",
                ),
                grid_template_columns="minmax(0, 1fr) 270px",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            rx.box(
                rx.image(
                    src=image,
                    alt=alt,
                    width="100%",
                    max_width="800px",
                    height="auto",
                    margin="0 auto",
                    display="block",
                ),
                width="100%",
                border_top=f"1px solid {LINE}",
                border_bottom=f"1px solid {LINE}",
                padding_y="1rem",
            ),
            rx.hstack(
                rx.icon("lightbulb", size=17, color=accent),
                rx.text(
                    takeaway,
                    size="1",
                    color=INK,
                    line_height="1.5",
                ),
                spacing="2",
                align="start",
                width="100%",
                background=tint,
                padding="0.75rem 0.9rem",
                border_radius="6px",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="1.25rem",
        width="100%",
    )


def model_name_token(name: str, accent: str, tint: str) -> rx.Component:
    return rx.text(
        name,
        size="1",
        color=accent,
        weight="bold",
        font_family=MONO,
        background=tint,
        border=f"1px solid {accent}22",
        border_radius="5px",
        padding="0.32rem 0.52rem",
        white_space="nowrap",
    )


def model_family_band(
    title: str,
    count: str,
    purpose: str,
    names: tuple[str, ...],
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.grid(
        rx.vstack(
            rx.hstack(
                rx.text(title, size="2", color=INK, weight="bold"),
                rx.badge(
                    count,
                    variant="soft",
                    color_scheme="teal" if accent == TEAL else "blue",
                ),
                spacing="2",
                align="center",
            ),
            rx.text(
                purpose,
                size="1",
                color=MUTED,
                line_height="1.45",
            ),
            spacing="1",
            align="start",
        ),
        rx.flex(
            *[
                model_name_token(name, accent, tint)
                for name in names
            ],
            wrap="wrap",
            gap="0.45rem",
            justify="start",
            width="100%",
        ),
        grid_template_columns="minmax(220px, 0.55fr) minmax(0, 1.45fr)",
        gap="1.5rem",
        align_items="center",
        padding="1rem 0",
        border_bottom=f"1px solid {LINE}",
        width="100%",
    )


def policy_branch(
    title: str,
    detail: str,
    status: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.icon("git-branch", size=18, color=accent),
            rx.text(title, size="2", color=INK, weight="bold"),
            rx.spacer(),
            rx.badge(
                status,
                variant="soft",
                color_scheme="teal" if accent == TEAL else "blue",
            ),
            width="100%",
            align="center",
        ),
        rx.text(
            detail,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        spacing="2",
        align="start",
        background=tint,
        border_left=f"3px solid {accent}",
        padding="1rem",
        min_height="112px",
        width="100%",
    )


def modelling_flow_step(
    icon: str,
    title: str,
    detail: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.icon(icon, size=20, color=accent),
            width="40px",
            height="40px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=tint,
            border_radius="7px",
        ),
        rx.text(
            title,
            size="2",
            color=INK,
            weight="bold",
            text_align="center",
        ),
        rx.text(
            detail,
            size="1",
            color=MUTED,
            line_height="1.4",
            text_align="center",
        ),
        spacing="2",
        align="center",
        justify="start",
        min_height="142px",
        width="100%",
        padding="0.85rem",
        border_top=f"2px solid {accent}",
        background=PANEL,
    )


def flow_arrow() -> rx.Component:
    return rx.box(
        rx.icon("arrow-right", size=18, color=MUTED),
        display="flex",
        align_items="center",
        justify_content="center",
        width="28px",
        flex_shrink="0",
    )


def formula_band(
    title: str,
    formula: str,
    baseline: str,
    detail: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(title, size="3", color=INK, weight="bold"),
            rx.spacer(),
            rx.badge(
                "LOCKED SYSTEM",
                variant="soft",
                color_scheme="teal" if accent == TEAL else "blue",
            ),
            width="100%",
            align="center",
        ),
        rx.text(
            formula,
            size="2",
            color=accent,
            font_family=MONO,
            weight="bold",
            line_height="1.6",
        ),
        rx.box(
            rx.vstack(
                rx.text(
                    "REFERENCE BASELINE",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                rx.text(
                    baseline,
                    size="1",
                    color=INK,
                    line_height="1.5",
                ),
                spacing="1",
                align="start",
            ),
            background=tint,
            padding="0.75rem 0.9rem",
            width="100%",
        ),
        rx.text(
            detail,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        spacing="3",
        align="start",
        padding="1.2rem",
        border_top=f"3px solid {accent}",
        border_bottom=f"1px solid {LINE}",
        min_height="250px",
        width="100%",
    )


def modelling_metric(
    title: str,
    formula: str,
    purpose: str,
    role: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.box(
        rx.grid(
            rx.vstack(
                rx.hstack(
                    rx.text(title, size="3", color=INK, weight="bold"),
                    rx.badge(
                        role,
                        variant="soft",
                        color_scheme="teal" if accent == TEAL else "blue",
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.text(
                    purpose,
                    size="1",
                    color=MUTED,
                    line_height="1.5",
                ),
                spacing="2",
                align="start",
            ),
            rx.box(
                rx.text(
                    formula,
                    size="2",
                    color=accent,
                    font_family=MONO,
                    weight="bold",
                    line_height="1.55",
                    text_align="center",
                ),
                background=tint,
                border_left=f"3px solid {accent}",
                padding="0.9rem 1rem",
                width="100%",
            ),
            grid_template_columns="minmax(0, 1.15fr) minmax(280px, 0.85fr)",
            gap="1.5rem",
            align_items="center",
            width="100%",
        ),
        border_bottom=f"1px solid {LINE}",
        padding="1rem 0",
        width="100%",
    )


def ensemble_formula_block(
    title: str,
    formula: str,
    objective: str,
    constraints: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(title, size="3", color=INK, weight="bold"),
            rx.spacer(),
            rx.badge(
                "FROZEN RECIPE",
                variant="soft",
                color_scheme="teal" if accent == TEAL else "blue",
            ),
            width="100%",
            align="center",
        ),
        rx.box(
            rx.text(
                formula,
                size="4",
                color=accent,
                font_family=MONO,
                weight="bold",
                line_height="1.75",
                white_space="pre-line",
                text_align="center",
            ),
            background=tint,
            border_top=f"3px solid {accent}",
            padding="1.4rem 1rem",
            width="100%",
        ),
        rx.grid(
            rx.vstack(
                rx.text(
                    "OPTIMISATION OBJECTIVE",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                rx.text(
                    objective,
                    size="1",
                    color=INK,
                    line_height="1.5",
                ),
                spacing="1",
                align="start",
            ),
            rx.vstack(
                rx.text(
                    "WEIGHT CONSTRAINTS",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                rx.text(
                    constraints,
                    size="1",
                    color=INK,
                    line_height="1.5",
                    font_family=MONO,
                ),
                spacing="1",
                align="start",
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        spacing="3",
        align="start",
        padding="1.2rem",
        border_bottom=f"1px solid {LINE}",
        width="100%",
    )


def baseline_level(
    number: str,
    title: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.grid(
        rx.box(
            rx.text(
                number,
                size="2",
                color=accent,
                weight="bold",
                font_family=MONO,
            ),
            width="34px",
            height="34px",
            display="flex",
            align_items="center",
            justify_content="center",
            background=tint,
            border_radius="50%",
        ),
        rx.text(title, size="3", color=INK, weight="bold"),
        grid_template_columns="34px minmax(0, 1fr)",
        gap="0.85rem",
        align_items="start",
        padding="0.7rem 0",
        border_bottom=f"1px solid {LINE}",
        width="100%",
    )


def baseline_explainer(
    title: str,
    definition: str,
    formula: str,
    levels: tuple[tuple[str, str], ...],
    final_note: str,
    accent: str,
    tint: str,
) -> rx.Component:
    return rx.box(
        rx.grid(
            rx.vstack(
                rx.text(title, size="4", color=INK, weight="bold"),
                rx.text(
                    definition,
                    size="2",
                    color=MUTED,
                    line_height="1.55",
                ),
                rx.box(
                    rx.text(
                        formula,
                        size="3",
                        color=accent,
                        font_family=MONO,
                        weight="bold",
                        line_height="1.65",
                        white_space="pre-line",
                        text_align="center",
                    ),
                    background=tint,
                    border_left=f"3px solid {accent}",
                    padding="1rem",
                    width="100%",
                ),
                rx.text(
                    final_note,
                    size="1",
                    color=INK,
                    line_height="1.5",
                    font_style="italic",
                ),
                spacing="3",
                align="start",
            ),
            rx.vstack(
                *[
                    baseline_level(number, level, accent, tint)
                    for number, level in levels
                ],
                spacing="0",
                align="start",
                width="100%",
            ),
            grid_template_columns="minmax(300px, 0.8fr) minmax(0, 1.2fr)",
            gap="2.25rem",
            align_items="start",
            width="100%",
        ),
        border_top=f"3px solid {accent}",
        border_bottom=f"1px solid {LINE}",
        padding="1.4rem",
        width="100%",
    )


def modelling_page() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text(
                        "MODELLING",
                        size="1",
                        color=TEAL,
                        weight="bold",
                    ),
                    rx.heading(
                        "Two learning problems developed through time",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        "Regression estimates a continuous fare adjustment. "
                        "Classification estimates a calibrated probability. "
                        "Both use expanding temporal folds and frozen ensemble "
                        "recipes.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                        max_width="760px",
                    ),
                    spacing="2",
                    align="start",
                ),
                rx.hstack(
                    modelling_stat("6", "development folds", TEAL),
                    rx.box(width="1px", height="58px", background=LINE),
                    modelling_stat("378", "registered training runs", BLUE),
                    spacing="5",
                    align="center",
                    justify="end",
                ),
                grid_template_columns="minmax(0, 1.55fr) minmax(340px, 0.85fr)",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border_bottom=f"1px solid {LINE}",
            padding="2rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "01 \u00b7 PROBLEM FORMULATION",
                    "What each model learns",
                    "Targets remain separate: one predicts a continuous "
                    "relative fare adjustment; one predicts an event "
                    "probability for the next canonical window.",
                ),
                rx.grid(
                    problem_definition(
                        "Regression",
                        "CONTINUOUS OUTPUT",
                        "y = log(observed fare / strictly-prior anchor)",
                        "predicted fare = anchor \u00d7 exp(predicted y)",
                        "Produces one VND estimate for every eligible "
                        "schedule-slot candidate at the selected booking session.",
                        TEAL,
                        TEAL_SOFT,
                        "chart-no-axes-combined",
                    ),
                    problem_definition(
                        "Classification",
                        "PROBABILITY OUTPUT",
                        "y = 1 when next-window fare \u2264 0.95 \u00d7 current fare",
                        "output = P(price drops by at least 5%)",
                        "Produces one calibrated probability before downstream "
                        "BUY or WAIT policy is applied.",
                        BLUE,
                        BLUE_SOFT,
                        "percent",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                ),
                spacing="4",
                align="start",
                width="100%",
            ),
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "02 \u00b7 TEMPORAL DEVELOPMENT",
                    "How training and validation move forward",
                    "Each fold expands legal training history, then validates "
                    "on a later week. Separate figures preserve the different "
                    "Fold 3 boundaries used by Regression and Classification.",
                ),
                modelling_temporal_figure(
                    "02A",
                    "Regression training expands before later validation",
                    "Development folds \u00b7 8 April\u201326 June 2026 \u00b7 each validation "
                    "week occurs strictly after its legal training history.",
                    "/modelling/regression_temporal_folds.png?v=20260731c",
                    "Paired training and validation row counts across six "
                    "expanding Regression folds",
                    "Future rows never enter fitting; unique out-of-fold "
                    "predictions feed Regression ensemble construction.",
                    [
                        ("6", "temporal folds"),
                        ("313,441", "unique OOF rows"),
                    ],
                    TEAL,
                    TEAL_SOFT,
                ),
                modelling_temporal_figure(
                    "02B",
                    "Classification preserves later validation transitions",
                    "Development folds \u00b7 8 April\u201326 June 2026 \u00b7 target "
                    "transitions mature before each validation week is scored.",
                    "/modelling/classification_temporal_folds.png?v=20260731c",
                    "Paired training and validation row counts across six "
                    "expanding Classification folds",
                    "Purged out-of-fold probabilities feed calibration and "
                    "ensemble construction; Fold 3 remains task-specific.",
                    [
                        ("6", "temporal folds"),
                        ("204,720", "unique OOF rows"),
                    ],
                    BLUE,
                    BLUE_SOFT,
                ),
                rx.vstack(
                    rx.text(
                        "CLASSIFICATION TRAINING-POLICY BRANCHES",
                        size="1",
                        color=BLUE,
                        weight="bold",
                    ),
                    rx.grid(
                        policy_branch(
                            "Within-period only",
                            "Cross-source transitions are excluded from "
                            "training while validation rows remain fixed.",
                            "LOCKED",
                            TEAL,
                            TEAL_SOFT,
                        ),
                        policy_branch(
                            "Include source bridge",
                            "Legally matched transitions crossing the source "
                            "boundary are retained as a development branch.",
                            "PARALLEL BRANCH",
                            BLUE,
                            BLUE_SOFT,
                        ),
                        columns="2",
                        spacing="4",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.icon("git-merge", size=17, color=MUTED),
                        rx.text(
                            "Both branches use the same 21 configurations and "
                            "the same validation rows. They do not create "
                            "separate BUY and WAIT models.",
                            size="1",
                            color=MUTED,
                            line_height="1.5",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    spacing="3",
                    align="start",
                    width="100%",
                    padding_top="0.75rem",
                    border_top=f"1px solid {LINE}",
                ),
                spacing="5",
                align="start",
                width="100%",
            ),
            background=PANEL,
            border_top=f"1px solid {LINE}",
            border_bottom=f"1px solid {LINE}",
            padding="1.5rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "03 \u00b7 MODEL REGISTRY",
                    "Which model configurations were trained",
                    "Every fold receives the same declared model registry. "
                    "Names below are frozen implementation names, not aliases.",
                ),
                rx.grid(
                    rx.vstack(
                        rx.hstack(
                            rx.text(
                                "REGRESSION",
                                size="1",
                                color=TEAL,
                                weight="bold",
                            ),
                            rx.spacer(),
                            rx.text(
                                "21 configurations/fold \u00b7 126 runs",
                                size="1",
                                color=MUTED,
                                font_family=MONO,
                            ),
                            width="100%",
                        ),
                        model_family_band(
                            "Tabular",
                            "8",
                            "Linear, tree, boosting and neural tabular learners.",
                            (
                                "Ridge",
                                "Elastic Net",
                                "Decision Tree",
                                "Random Forest",
                                "HistGBM",
                                "XGBoost",
                                "CatBoost",
                                "MLP",
                            ),
                            TEAL,
                            TEAL_SOFT,
                        ),
                        model_family_band(
                            "Recurrent",
                            "12",
                            "Four architectures at three legal history lengths.",
                            (
                                "RNN \u00b7 L7/L14/L21",
                                "GRU \u00b7 L7/L14/L21",
                                "LSTM \u00b7 L7/L14/L21",
                                "BiLSTM \u00b7 L7/L14/L21",
                            ),
                            TEAL,
                            TEAL_SOFT,
                        ),
                        model_family_band(
                            "Diagnostic",
                            "1",
                            "Foundation diagnostic retained outside winner selection.",
                            ("Chronos-2",),
                            TEAL,
                            TEAL_SOFT,
                        ),
                        spacing="0",
                        align="start",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.text(
                                "CLASSIFICATION",
                                size="1",
                                color=BLUE,
                                weight="bold",
                            ),
                            rx.spacer(),
                            rx.text(
                                "21 configurations/policy/fold \u00b7 252 runs",
                                size="1",
                                color=MUTED,
                                font_family=MONO,
                            ),
                            width="100%",
                        ),
                        model_family_band(
                            "Tabular",
                            "9",
                            "Linear, tree, boosting and neural classifiers.",
                            (
                                "Logistic Regression",
                                "Linear SVM",
                                "Decision Tree",
                                "Random Forest",
                                "HistGBM",
                                "XGBoost",
                                "CatBoost",
                                "Delta CatBoost",
                                "MLP",
                            ),
                            BLUE,
                            BLUE_SOFT,
                        ),
                        model_family_band(
                            "Recurrent",
                            "12",
                            "Four architectures at three legal history lengths.",
                            (
                                "RNN \u00b7 L7/L14/L21",
                                "GRU \u00b7 L7/L14/L21",
                                "LSTM \u00b7 L7/L14/L21",
                                "BiLSTM \u00b7 L7/L14/L21",
                            ),
                            BLUE,
                            BLUE_SOFT,
                        ),
                        spacing="0",
                        align="start",
                        width="100%",
                    ),
                    columns="2",
                    spacing="5",
                    align_items="start",
                    width="100%",
                ),
                spacing="5",
                align="start",
                width="100%",
            ),
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "04 \u00b7 MODELLING PIPELINES",
                    "How recurrent and tabular paths share one population",
                    "Separate scientific workflow figures show routing, "
                    "fallback, calibration and output construction without "
                    "mixing either task.",
                ),
                modelling_figure(
                    "/modelling/previews/regression_pipeline_vertical_preview.png?v=20260730d",
                    "Regression modelling pipeline showing tabular and recurrent "
                    "routing, fallback, ensemble and fare reconstruction",
                    "Regression learns routed log-ratio residuals; the final "
                    "continuous fare is reconstructed against its strictly-prior anchor.",
                    TEAL,
                    "980px",
                ),
                modelling_figure(
                    "/modelling/previews/classification_pipeline_vertical_preview.png?v=20260730d",
                    "Classification modelling pipeline showing tabular and "
                    "recurrent routing, calibration, hierarchy and probability output",
                    "Classification produces one calibrated drop probability. "
                    "BUY and WAIT remain downstream policy actions, not separate models.",
                    BLUE,
                    "980px",
                ),
                spacing="5",
                align="start",
                width="100%",
            ),
            background=PANEL,
            border_top=f"1px solid {LINE}",
            border_bottom=f"1px solid {LINE}",
            padding="1.5rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "05 \u00b7 DEVELOPMENT SELECTION CRITERIA",
                    "Which metrics define a useful model",
                    "These criteria guide development selection only. Metric "
                    "values and final performance remain under Evaluation.",
                ),
                rx.grid(
                    rx.vstack(
                        rx.text(
                            "REGRESSION",
                            size="1",
                            color=TEAL,
                            weight="bold",
                        ),
                        modelling_metric(
                            "MAPE",
                            "mean(|actual \u2212 predicted| / actual)",
                            "Measures relative fare error across schedule slots.",
                            "FARE ACCURACY",
                            TEAL,
                            TEAL_SOFT,
                        ),
                        modelling_metric(
                            "Cheapest-flight regret",
                            "mean(fare chosen \u2212 true cheapest fare)",
                            "Measures VND cost when ranking selects a non-cheapest flight.",
                            "RANKING UTILITY",
                            TEAL,
                            TEAL_SOFT,
                        ),
                        spacing="0",
                        align="start",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text(
                            "CLASSIFICATION",
                            size="1",
                            color=BLUE,
                            weight="bold",
                        ),
                        modelling_metric(
                            "Brier score",
                            "mean((P(drop) \u2212 observed event)\u00b2)",
                            "Measures calibration and accuracy of event probabilities.",
                            "PROBABILITY QUALITY",
                            BLUE,
                            BLUE_SOFT,
                        ),
                        modelling_metric(
                            "Policy regret",
                            "mean(VND cost of chosen action)",
                            "Measures economic consequence after the frozen "
                            "probability threshold maps to BUY or WAIT.",
                            "DECISION UTILITY",
                            BLUE,
                            BLUE_SOFT,
                        ),
                        spacing="0",
                        align="start",
                        width="100%",
                    ),
                    columns="2",
                    spacing="5",
                    align_items="start",
                    width="100%",
                ),
                spacing="4",
                align="start",
                width="100%",
            ),
            background="#F7FAFA",
            border_bottom=f"1px solid {LINE}",
            padding="1.5rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "06 \u00b7 ENSEMBLE CONSTRUCTION",
                    "How candidates become one prediction",
                    "SLSQP estimates convex weights from development out-of-fold "
                    "predictions. Every weight is non-negative and all weights sum to one.",
                ),
                rx.grid(
                    ensemble_formula_block(
                        "Regression",
                        "r\u0302\u1d62 = \u03a3\u2c7c w\u2c7c r\u0302\u1d62\u2c7c\nPredicted fare\u1d62 = anchor\u1d62 \u00d7 exp(clip(r\u0302\u1d62))",
                        "Minimise mean squared error on development out-of-fold "
                        "log-ratio residuals.",
                        "w\u2c7c \u2265 0     \u03a3\u2c7cw\u2c7c = 1",
                        TEAL,
                        TEAL_SOFT,
                    ),
                    ensemble_formula_block(
                        "Classification",
                        "p\u0303\u1d62\u2c7c = sigmoid(a\u2c7cs\u1d62\u2c7c + b\u2c7c)\nP\u0302\u1d62(drop) = \u03a3\u2c7c w\u2c7c p\u0303\u1d62\u2c7c",
                        "Minimise out-of-fold Brier loss plus 0.01 shrinkage "
                        "towards equal weights.",
                        "w\u2c7c \u2265 0     \u03a3\u2c7cw\u2c7c = 1",
                        BLUE,
                        BLUE_SOFT,
                    ),
                    columns="2",
                    spacing="4",
                    align_items="start",
                    width="100%",
                ),
                spacing="4",
                align="start",
                width="100%",
            ),
            background=PANEL,
            border_top=f"1px solid {LINE}",
            border_bottom=f"1px solid {LINE}",
            padding="1.5rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                modelling_section_heading(
                    "07 \u00b7 REFERENCE BASELINES",
                    "What each ensemble must improve upon",
                    "Baselines are legal, point-in-time reference systems. "
                    "They remain explicit comparisons and never use future information.",
                ),
                baseline_explainer(
                    "Regression baseline",
                    "The baseline predicts the strictly-prior anchor unchanged. "
                    "It searches the latest completed prior batch from most "
                    "specific support to broad fallback support.",
                    "baseline fare = strictly-prior anchor\nbaseline log-ratio = 0",
                    (
                        (
                            "1",
                            "Route + airline + departure period + DUD",
                        ),
                        (
                            "2",
                            "Route + airline + DUD",
                        ),
                        (
                            "3",
                            "Route + DUD",
                        ),
                        (
                            "4",
                            "Airline + DUD",
                        ),
                        (
                            "5",
                            "Global DUD",
                        ),
                        (
                            "6",
                            "Global prior-batch median",
                        ),
                    ),
                    "A zero predicted residual reproduces this baseline exactly.",
                    TEAL,
                    TEAL_SOFT,
                ),
                baseline_explainer(
                    "Classification baseline",
                    "The baseline estimates event probability through fixed "
                    "hierarchical shrinkage. Sparse groups borrow strength from "
                    "their broader parent rather than producing unstable rates.",
                    "global \u2192 transition \u2192 route + airline + transition",
                    (
                        (
                            "1",
                            "Global event rate",
                        ),
                        (
                            "2",
                            "Transition probability",
                        ),
                        (
                            "3",
                            "Route + airline + transition probability",
                        ),
                    ),
                    "The hierarchy probability joins calibrated model "
                    "probabilities as one declared ensemble component.",
                    BLUE,
                    BLUE_SOFT,
                ),
                spacing="5",
                align="start",
                width="100%",
            ),
            background="#F7FAFA",
            border_top=f"1px solid {LINE}",
            padding="1.5rem",
            width="100%",
        ),
        rx.text(
            "Metric values, component weights, calibration curves, feature "
            "dependence, ablation and final temporal results are reported under Evaluation.",
            size="1",
            color=MUTED,
            font_style="italic",
        ),
        spacing="6",
        width="100%",
    )


def _evaluation_section(
    number: str,
    title: str,
    description: str,
) -> rx.Component:
    return rx.vstack(
        rx.text(number, size="1", color=TEAL, weight="bold", font_family=MONO),
        rx.heading(title, size="6", color=INK, letter_spacing="0"),
        rx.text(
            description,
            size="2",
            color=MUTED,
            line_height="1.55",
            max_width="780px",
        ),
        spacing="1",
        align="start",
        width="100%",
        border_bottom=f"1px solid {LINE}",
        padding_bottom="0.9rem",
    )


def _evaluation_figure(
    number: str,
    scope: str,
    title: str,
    description: str,
    image: str,
    alt: str,
    takeaway: str,
    stats: list[tuple[str, str]],
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.grid(
                rx.vstack(
                    rx.text(
                        f"FIGURE {number} \u00b7 {scope}",
                        size="1",
                        color=TEAL,
                        weight="bold",
                        font_family=MONO,
                    ),
                    rx.heading(
                        title,
                        size="5",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        description,
                        size="2",
                        color=MUTED,
                        line_height="1.55",
                        max_width="720px",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.hstack(
                    *[
                        evidence_stat(value, label, BLUE)
                        for value, label in stats
                    ],
                    spacing="4",
                    align="center",
                    justify="end",
                    width="100%",
                ),
                grid_template_columns="minmax(0, 1fr) 270px",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            rx.box(
                rx.image(
                    src=image,
                    alt=alt,
                    width="100%",
                    max_width="800px",
                    height="auto",
                    margin="0 auto",
                    display="block",
                ),
                width="100%",
                border_top=f"1px solid {LINE}",
                border_bottom=f"1px solid {LINE}",
                padding_y="1rem",
            ),
            rx.hstack(
                rx.icon("lightbulb", size=17, color=TEAL),
                rx.text(
                    takeaway,
                    size="1",
                    color=INK,
                    line_height="1.5",
                ),
                spacing="2",
                align="start",
                width="100%",
                background=TEAL_SOFT,
                padding="0.75rem 0.9rem",
                border_radius="6px",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="1.25rem",
        width="100%",
    )


def _evaluation_table(
    title: str,
    description: str,
    headers: list[str],
    rows: list[list[str]],
) -> rx.Component:
    header_style = {
        "background": "#F2F6F6",
        "color": INK,
        "font_weight": "700",
        "padding": "0.7rem 0.8rem",
        "border_bottom": f"1px solid {LINE}",
        "white_space": "nowrap",
    }
    cell_style = {
        "padding": "0.68rem 0.8rem",
        "border_bottom": f"1px solid {LINE}",
        "color": INK,
        "font_variant_numeric": "tabular-nums",
        "vertical_align": "top",
    }
    return rx.box(
        rx.vstack(
            rx.heading(title, size="4", color=INK, letter_spacing="0"),
            rx.text(
                description,
                size="1",
                color=MUTED,
                line_height="1.5",
            ),
            rx.box(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            *[
                                rx.table.column_header_cell(
                                    header,
                                    **header_style,
                                )
                                for header in headers
                            ]
                        )
                    ),
                    rx.table.body(
                        *[
                            rx.table.row(
                                *[
                                    rx.table.cell(value, **cell_style)
                                    for value in row
                                ]
                            )
                            for row in rows
                        ]
                    ),
                    variant="surface",
                    size="2",
                    width="100%",
                    background=PANEL,
                ),
                width="100%",
                overflow_x="auto",
                border=f"1px solid {LINE}",
                border_radius="6px",
            ),
            spacing="2",
            align="start",
            width="100%",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="1.2rem",
        width="100%",
    )


def evaluation_page() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text("EVALUATION", size="1", color=TEAL, weight="bold"),
                    rx.heading(
                        "Frozen evidence from development to final evaluation",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                    ),
                    rx.text(
                        "Candidate selection, temporal stability, model "
                        "dependence, interpolation, and final held-out evidence "
                        "are reported without retraining or reselection.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                        max_width="760px",
                    ),
                    spacing="2",
                    align="start",
                ),
                rx.vstack(
                    rx.text(
                        "FROZEN EVALUATION PROTOCOL",
                        size="1",
                        color=BLUE,
                        weight="bold",
                    ),
                    rx.text(
                        "No retraining \u00b7 No recalibration \u00b7 No threshold search",
                        size="2",
                        color=INK,
                        weight="bold",
                        text_align="right",
                    ),
                    spacing="1",
                    align="end",
                ),
                grid_template_columns="minmax(0, 1fr) 330px",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            padding="1.5rem",
            width="100%",
        ),
        rx.grid(
            evidence_stat("8 Apr\u201326 Jun", "development", TEAL),
            evidence_stat("27 Jun\u201310 Jul", "locked validation", BLUE),
            evidence_stat("16\u201328 Jul", "final temporal evaluation", AMBER),
            columns="3",
            spacing="4",
            width="100%",
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            padding="1.1rem",
        ),
        _evaluation_section(
            "01 \u00b7 DEVELOPMENT SELECTION",
            "Why the locked ensembles were selected",
            "Six rolling-origin folds compare base candidates, baselines, "
            "and frozen combinations using only later-period validation rows.",
        ),
        _evaluation_figure(
            "01",
            "DEVELOPMENT OOF DESCRIPTIVE EVIDENCE",
            "Ensemble leads regression candidates",
            "Out-of-fold MAPE and cheapest-flight regret compare 20 base "
            "configurations with the prior-anchor baseline and locked ensemble.",
            "/evaluation/01_regression_candidate_landscape.png?v=20260731b",
            "Regression development candidate landscape by MAPE and regret",
            "The locked ensemble occupies the strongest joint region without "
            "using final temporal evaluation data.",
            [("20", "base candidates"), ("313,441", "unique OOF rows")],
        ),
        _evaluation_figure(
            "02",
            "DEVELOPMENT OOF DESCRIPTIVE EVIDENCE",
            "Ensemble improves both classification objectives",
            "Frozen calibrated component probabilities are compared by Brier "
            "score and decision regret at the locked 0.30 threshold.",
            "/evaluation/02_classification_candidate_landscape.png?v=20260731b",
            "Classification development candidate landscape by Brier score and regret",
            "Candidate evidence remains descriptive because calibration and "
            "weights were selected from development OOF predictions.",
            [("21", "base candidates"), ("204,720", "unique OOF rows")],
        ),
        _evaluation_figure(
            "03",
            "SIX ROLLING-ORIGIN FOLDS",
            "Both systems remain stable across six folds",
            "Model-to-baseline ratios below 1.00 indicate improvement on "
            "later-period validation rows.",
            "/evaluation/03_six_fold_stability.png?v=20260731a",
            "Regression and classification metric ratios over six development folds",
            "The frozen recipes improve predictive and decision metrics across "
            "all six temporal development folds.",
            [("6", "temporal folds"), ("1.00", "baseline reference")],
        ),
        _evaluation_figure(
            "04",
            "FROZEN REGRESSION RECIPE",
            "Regression weights combine eight components",
            "Non-negative convex weights sum to one; CatBoost remains the "
            "locked fallback for rows without eligible recurrent history.",
            "/evaluation/04_regression_ensemble_weights.png?v=20260731a",
            "Frozen regression ensemble weights",
            "Weight is distributed across tabular and recurrent evidence "
            "rather than assigned to one dominant component.",
            [("8", "weighted components"), ("1.00", "weight sum")],
        ),
        _evaluation_figure(
            "05",
            "FROZEN CLASSIFICATION RECIPE",
            "Classification weights retain diverse evidence",
            "Shrinkage-constrained simplex weights combine calibrated model "
            "probabilities and the hierarchical probability component.",
            "/evaluation/05_classification_ensemble_weights.png?v=20260731a",
            "Frozen classification ensemble weights",
            "Positive weights preserve complementary evidence while zero-weight "
            "candidates remain recorded in provenance.",
            [("16", "weighted components"), ("0.30", "policy threshold")],
        ),
        _evaluation_section(
            "02 \u00b7 MODEL DEPENDENCE AND INTERPOLATION",
            "What the systems depend on",
            "Post-fit diagnostics test grouped feature reliance and the "
            "deterministic off-grid interpolation module.",
        ),
        _evaluation_figure(
            "06",
            "DEVELOPMENT DIAGNOSTIC",
            "No feature group monopolises prediction",
            "Grouped permutation keeps correlated fields together and measures "
            "performance change after repeated development-data perturbations.",
            "/evaluation/06_grouped_feature_dependence.png?v=20260731a",
            "Grouped permutation dependence for regression and classification",
            "Both systems use multiple feature families; no single field-level "
            "importance is presented as causal evidence.",
            [("46.1%", "largest regression share"), ("41.6%", "largest classification share")],
        ),
        _evaluation_figure(
            "08",
            "EVALUATION-ONLY DETERMINISTIC DERIVATION",
            "Interpolation beats nearest-window carry",
            "Nine interior windows replay frozen log-price interpolation. "
            "Paired slot-level bootstrap compares it with nearest-window carry.",
            "/evaluation/08_lobo_interpolation.png?v=20260731a",
            "Leave-one-booking-window-out interpolation skill with confidence intervals",
            "All nine interior windows retain positive paired skill without "
            "model fitting or changes to the frozen recipe.",
            [("9", "interior windows"), ("5,000", "cluster replicates")],
        ),
        _evaluation_section(
            "03 \u00b7 FINAL TEMPORAL EVALUATION",
            "How frozen systems perform on unseen later data",
            "Predictions use pre-evaluation artifacts; all confidence intervals "
            "are pointwise 95% percentile intervals clustered by schedule slot.",
        ),
        _evaluation_figure(
            "07",
            "FINAL TEMPORAL EVALUATION",
            "Final fares track booking-window profiles",
            "Observed and locked-ensemble fares are "
            "compared over the eleven canonical booking windows.",
            "/evaluation/07_final_fare_profile.png?v=20260805a",
            "Final temporal fare profile by days until departure",
            "The frozen regression ensemble follows observed fare structure "
            "more closely than the anchor baseline.",
            [("129,362", "evaluation rows"), ("11", "booking windows")],
        ),
        _evaluation_figure(
            "09",
            "FINAL TEMPORAL EVALUATION",
            "Classification probabilities improve calibration",
            "Reliability curves use shared fixed bins; the histogram shows "
            "how many predictions support each probability region.",
            "/evaluation/09_probability_calibration.png?v=20260805a",
            "Probability calibration and prediction histogram",
            "The locked ensemble improves Brier score and log loss while "
            "retaining explicit probability support information.",
            [("63,109", "evaluation rows"), ("11.0%", "observed DROP rate")],
        ),
        _evaluation_figure(
            "10",
            "LOCKED THRESHOLD 0.30",
            "Outcome-action matrix shows policy trade-off",
            "Rows are observed DROP outcomes; columns are downstream BUY/WAIT "
            "actions. No actual BUY/WAIT label is claimed.",
            "/evaluation/10_outcome_action_matrix.png?v=20260731a",
            "Observed price outcome by downstream policy action",
            "The threshold deliberately trades additional WAIT actions for "
            "greater DROP capture under the frozen decision policy.",
            [("0.30", "locked threshold"), ("11.0%", "actual DROP share")],
        ),
        _evaluation_section(
            "04 \u00b7 EVIDENCE TABLES AND REPRODUCIBILITY",
            "Auditable metrics and derivation records",
            "Tables preserve locked validation, final temporal metrics, and "
            "reporting-only derivation controls.",
        ),
        _evaluation_table(
            "Locked validation summary",
            "Two later held-out blocks checked stability before final refit.",
            ["Task", "Block", "Metric", "Locked", "Baseline", "Regret", "Baseline regret", "Status"],
            [
                ["Regression", "Locked Validation 1", "MAPE", "7.68%", "12.75%", "60,416 VND", "148,701 VND", "PASS"],
                ["Classification", "Locked Validation 1", "Brier", "0.2260", "0.2219", "61,243 VND", "67,486 VND", "UTILITY PASS; BRIER WITHIN 5% GUARDRAIL"],
                ["Regression", "Locked Validation 2", "MAPE", "8.76%", "13.79%", "60,367 VND", "153,293 VND", "PASS"],
                ["Classification", "Locked Validation 2", "Brier", "0.1481", "0.1589", "44,951 VND", "57,301 VND", "PASS"],
            ],
        ),
        _evaluation_table(
            "Final temporal metrics",
            "Frozen systems are compared with their task-specific baselines on 16\u201328 July 2026.",
            ["Task", "Metric", "Locked system", "Baseline"],
            [
                ["Regression", "MAPE", "7.26%", "13.11%"],
                ["Regression", "Cheapest-flight regret", "50,002 VND", "139,641 VND"],
                ["Regression", "R-squared", "0.852", "0.606"],
                ["Classification", "Brier score", "0.1042", "0.1173"],
                ["Classification", "Decision regret", "29,390 VND", "42,001 VND"],
                ["Classification", "Log loss", "0.3627", "0.3977"],
                ["Classification", "Average precision", "0.2363", "0.1388"],
            ],
        ),
        _evaluation_table(
            "Integrity, bootstrap and independent reproduction",
            "Reporting evidence remains traceable to frozen source and artifact hashes.",
            ["Evidence", "Value", "Status"],
            [
                ["Final temporal row scope", "129,362 regression rows; 63,109 classification rows", "PASS"],
                ["Bootstrap design", "5,000 paired cluster replicates; pointwise percentile 95%; seed 20260731", "PASS"],
                ["Cluster key", "schedule_slot_id", "PASS"],
                ["LOBO derivation", "Deterministic evaluation-only replay; no model fitting; nine interior windows", "PASS"],
                ["Independent reproduction", "Frozen evaluator reproduced metrics independently", "PASS"],
                ["Frozen methodology", "No retraining, reselection, recalibration, or threshold search", "PASS"],
            ],
        ),
        rx.text(
            "Derived evidence manifest records source hashes, artifact hashes, "
            "script version, row counts, DUD/window, seed, replicate count, "
            "cluster key, and confidence-interval method.",
            size="1",
            color=MUTED,
            font_style="italic",
        ),
        spacing="6",
        width="100%",
    )


def _limitations_section(
    number: str,
    title: str,
    description: str,
) -> rx.Component:
    return rx.vstack(
        rx.text(
            number,
            size="1",
            color=TEAL,
            weight="bold",
            font_family=MONO,
        ),
        rx.heading(
            title,
            size="6",
            color=INK,
            letter_spacing="0",
        ),
        rx.text(
            description,
            size="2",
            color=MUTED,
            line_height="1.55",
            max_width="820px",
        ),
        spacing="1",
        align="start",
        width="100%",
        border_bottom=f"1px solid {LINE}",
        padding_bottom="0.9rem",
    )


def _scope_badge(label: str, tone: str) -> rx.Component:
    colors = {
        "supported": ("teal", TEAL_SOFT),
        "bounded": ("amber", "#FFF5E5"),
        "fixed": ("blue", BLUE_SOFT),
    }
    color_scheme, background = colors[tone]
    return rx.badge(
        label,
        color_scheme=color_scheme,
        variant="soft",
        size="1",
        background=background,
        white_space="nowrap",
    )


def _scope_row(
    title: str,
    status: str,
    tone: str,
    supported: str,
    boundary: str,
) -> rx.Component:
    return rx.grid(
        rx.vstack(
            rx.text(title, size="2", color=INK, weight="bold"),
            _scope_badge(status, tone),
            spacing="1",
            align="start",
        ),
        rx.text(
            supported,
            size="1",
            color=INK,
            line_height="1.5",
        ),
        rx.text(
            boundary,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        grid_template_columns="190px minmax(0, 1fr) minmax(0, 1fr)",
        gap="1.25rem",
        align_items="start",
        width="100%",
        padding="1rem",
        border_bottom=f"1px solid {LINE}",
    )


def _limitation_row(
    icon: str,
    title: str,
    limitation: str,
    consequence: str,
    safeguard: str,
    residual: str,
) -> rx.Component:
    return rx.grid(
        rx.hstack(
            rx.icon(icon, size=18, color=TEAL),
            rx.text(
                title,
                size="2",
                color=INK,
                weight="bold",
            ),
            spacing="2",
            align="start",
        ),
        rx.text(
            limitation,
            size="1",
            color=INK,
            line_height="1.5",
        ),
        rx.text(
            consequence,
            size="1",
            color=INK,
            line_height="1.5",
        ),
        rx.text(
            safeguard,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        rx.text(
            residual,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        grid_template_columns=(
            "180px minmax(0, 1fr) minmax(0, 0.9fr) "
            "minmax(0, 1fr) minmax(0, 0.9fr)"
        ),
        gap="1rem",
        align_items="start",
        width="100%",
        padding="1rem",
        border_bottom=f"1px solid {LINE}",
    )


def _request_boundary_row(
    condition: str,
    output: str,
    explanation: str,
    tone: str,
) -> rx.Component:
    return rx.grid(
        rx.text(
            condition,
            size="2",
            color=INK,
            weight="bold",
            line_height="1.45",
        ),
        _scope_badge(output, tone),
        rx.text(
            explanation,
            size="1",
            color=MUTED,
            line_height="1.5",
        ),
        grid_template_columns="minmax(0, 1.1fr) 210px minmax(0, 1fr)",
        gap="1.25rem",
        align_items="center",
        width="100%",
        padding="0.9rem 1rem",
        border_bottom=f"1px solid {LINE}",
    )


def _plain_limit_item(
    icon: str,
    text: str,
    accent: str = MUTED,
) -> rx.Component:
    return rx.hstack(
        rx.icon(icon, size=17, color=accent),
        rx.text(
            text,
            size="1",
            color=INK,
            line_height="1.5",
        ),
        spacing="2",
        align="start",
        width="100%",
    )


def limitations_page() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.grid(
                rx.vstack(
                    rx.text(
                        "LIMITATIONS",
                        size="1",
                        color=TEAL,
                        weight="bold",
                    ),
                    rx.heading(
                        "Evidence boundaries and known failure modes",
                        size="8",
                        color=INK,
                        letter_spacing="0",
                        line_height="1.12",
                    ),
                    rx.text(
                        "SkyFare is a decision-support prototype. This page "
                        "states where evidence is supported, where outputs are "
                        "bounded, and how unsupported requests are handled.",
                        size="3",
                        color=MUTED,
                        line_height="1.6",
                        max_width="760px",
                    ),
                    spacing="2",
                    align="start",
                ),
                rx.vstack(
                    rx.text(
                        "USE BOUNDARY",
                        size="1",
                        color=AMBER,
                        weight="bold",
                    ),
                    rx.text(
                        "Prediction, not a guaranteed fare or saving",
                        size="2",
                        color=INK,
                        weight="bold",
                        text_align="right",
                        line_height="1.45",
                    ),
                    rx.text(
                        "One-way \u00b7 1 adult \u00b7 Economy \u00b7 Standard fare",
                        size="1",
                        color=MUTED,
                        text_align="right",
                    ),
                    spacing="1",
                    align="end",
                    border_left=f"4px solid {AMBER}",
                    padding_left="1rem",
                ),
                grid_template_columns="minmax(0, 1fr) 340px",
                gap="2rem",
                align_items="center",
                width="100%",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            padding="1.5rem",
            width="100%",
        ),
        _limitations_section(
            "01 \u00b7 OPERATING BOUNDARY",
            "What SkyFare supports",
            "Outputs remain tied to the observed market, canonical booking "
            "windows, and schedule candidates known before query time.",
        ),
        rx.box(
            rx.grid(
                rx.text("CAPABILITY", size="1", color=MUTED, weight="bold"),
                rx.text(
                    "SUPPORTED USE",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                rx.text(
                    "BOUNDARY",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                grid_template_columns="190px minmax(0, 1fr) minmax(0, 1fr)",
                gap="1.25rem",
                width="100%",
                padding="0.75rem 1rem",
                background="#F2F6F6",
                border_bottom=f"1px solid {LINE}",
            ),
            _scope_row(
                "Regression",
                "SUPPORTED + BOUNDED",
                "supported",
                "Direct fare estimates at the 11 canonical DUD windows. "
                "Interior off-grid dates receive tested log-price interpolation.",
                "Outside DUD 1\u201360, missing schedule candidates, and endpoint "
                "extrapolation receive no fare estimate.",
            ),
            _scope_row(
                "Classification",
                "ON-GRID ONLY",
                "bounded",
                "P(DROP5) and BUY/WAIT apply only to the next canonical "
                "booking-window transition.",
                "Off-grid dates receive no probability or booking guidance. "
                "Missing future observations are never labelled NO DROP.",
            ),
            _scope_row(
                "Market coverage",
                "FIXED COVERAGE",
                "fixed",
                "Twenty domestic routes, five airlines, and 72 supported "
                "route-airline pairs.",
                "Candidate schedule must be available before query time. "
                "Unsupported routes or airlines receive no prediction.",
            ),
            _scope_row(
                "Fare product",
                "FIXED PRODUCT",
                "fixed",
                "One-way travel for one adult in economy using the standard-fare "
                "population.",
                "Baggage, refundability, change fees, loyalty benefits, and "
                "other fare-family conditions are not modelled.",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            overflow="hidden",
            width="100%",
        ),
        _limitations_section(
            "02 \u00b7 KNOWN LIMITATIONS",
            "What may fail, why it matters, and what remains controlled",
            "Each limitation is connected to its product consequence, current "
            "safeguard, and remaining uncertainty.",
        ),
        rx.box(
            rx.grid(
                rx.text("AREA", size="1", color=MUTED, weight="bold"),
                rx.text("LIMITATION", size="1", color=MUTED, weight="bold"),
                rx.text("WHY IT MATTERS", size="1", color=MUTED, weight="bold"),
                rx.text(
                    "CURRENT SAFEGUARD",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                rx.text(
                    "REMAINING RISK",
                    size="1",
                    color=MUTED,
                    weight="bold",
                ),
                grid_template_columns=(
                    "180px minmax(0, 1fr) minmax(0, 0.9fr) "
                    "minmax(0, 1fr) minmax(0, 0.9fr)"
                ),
                gap="1rem",
                width="100%",
                padding="0.75rem 1rem",
                background="#F2F6F6",
                border_bottom=f"1px solid {LINE}",
            ),
            _limitation_row(
                "eye-off",
                "Public data visibility",
                "Seat inventory, load factor, promotions, booking velocity, "
                "and airline revenue-management controls are not observed.",
                "Abrupt price changes may have no visible precursor.",
                "Strictly-prior market context, legal price history, and a "
                "hierarchical anchor reduce level drift.",
                "The system cannot anticipate or explain every fare movement.",
            ),
            _limitation_row(
                "circle-help",
                "Target observability",
                "Classification scoring uses only targets that are both mature "
                "and observed at the next canonical window.",
                "A missing re-observation has an unknown outcome.",
                "Immature and unobserved targets stay outside metric "
                "denominators and never become NO DROP labels.",
                "Reported metrics remain conditional on observable transitions.",
            ),
            _limitation_row(
                "route",
                "Coverage transfer",
                "Training evidence covers the supported domestic "
                "route-airline population and standard fare product.",
                "Performance may not transfer to unseen markets or fare rules.",
                "Serving filters route, airline, date, and departure-time "
                "candidates against the prior schedule cache.",
                "Expansion requires new collection and temporal validation.",
            ),
            _limitation_row(
                "clock-3",
                "Freshness and drift",
                "Market conditions and fare levels change after the frozen "
                "training and evaluation periods.",
                "A previously strong model may weaken as the market moves.",
                "Rolling-origin development, later temporal evaluation, model "
                "versioning, data cutoff, and as-of disclosure.",
                "No historical evaluation guarantees future performance.",
            ),
            _limitation_row(
                "split",
                "Decision policy",
                "A fixed 0.30 probability threshold converts classification "
                "output into BUY or WAIT guidance.",
                "False WAIT and missed DROP decisions remain possible.",
                "Probability calibration, Brier score, decision regret, and "
                "threshold metrics are reported together under Evaluation.",
                "Guidance supports a user decision; it does not automate one.",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            overflow="hidden",
            width="100%",
        ),
        rx.callout(
            "Thresholded DROP recognition remains limited. Precision, recall, "
            "F1, class shares, and confusion evidence are reported in the "
            "Evaluation tab rather than hidden behind decision regret.",
            icon="info",
            color_scheme="amber",
            variant="surface",
            width="100%",
        ),
        _limitations_section(
            "03 \u00b7 REQUEST HANDLING",
            "Unsupported cases fail visibly",
            "The interface changes the returned evidence instead of silently "
            "substituting a different task.",
        ),
        rx.box(
            rx.grid(
                rx.text("REQUEST CONDITION", size="1", color=MUTED, weight="bold"),
                rx.text("RETURNED OUTPUT", size="1", color=MUTED, weight="bold"),
                rx.text("SERVING RULE", size="1", color=MUTED, weight="bold"),
                grid_template_columns="minmax(0, 1.1fr) 210px minmax(0, 1fr)",
                gap="1.25rem",
                width="100%",
                padding="0.75rem 1rem",
                background="#F2F6F6",
                border_bottom=f"1px solid {LINE}",
            ),
            _request_boundary_row(
                "Known candidate at a canonical DUD",
                "FARE + BUY/WAIT",
                "Regression and classification contracts are both supported.",
                "supported",
            ),
            _request_boundary_row(
                "Known candidate at an interior off-grid DUD",
                "FARE ONLY",
                "Regression uses frozen interpolation; classification remains "
                "unavailable.",
                "bounded",
            ),
            _request_boundary_row(
                "DUD outside 1\u201360 or unsupported route-airline",
                "NO PREDICTION",
                "The app reports an explicit scope error instead of filling "
                "from another market.",
                "fixed",
            ),
            _request_boundary_row(
                "Source batch incomplete or stale",
                "AS-OF DISCLOSURE",
                "The result is historical evidence and must not be presented "
                "as a fresh market quote.",
                "bounded",
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            overflow="hidden",
            width="100%",
        ),
        _limitations_section(
            "04 \u00b7 NON-CLAIMS",
            "What SkyFare does not promise",
            "These statements prevent a decision-support prototype from being "
            "interpreted as a live booking guarantee.",
        ),
        rx.box(
            rx.grid(
                rx.vstack(
                    _plain_limit_item(
                        "x",
                        "Guaranteed future fare or guaranteed saving.",
                        CORAL,
                    ),
                    _plain_limit_item(
                        "x",
                        "Causal explanation of airline pricing decisions.",
                        CORAL,
                    ),
                    _plain_limit_item(
                        "x",
                        "Live seat availability or sold-out status.",
                        CORAL,
                    ),
                    spacing="3",
                    align="start",
                    width="100%",
                ),
                rx.vstack(
                    _plain_limit_item(
                        "x",
                        "BUY/WAIT guidance for off-grid booking dates.",
                        CORAL,
                    ),
                    _plain_limit_item(
                        "x",
                        "Coverage beyond supported routes, airlines, and fare scope.",
                        CORAL,
                    ),
                    _plain_limit_item(
                        "x",
                        "Ticket purchase, payment, or booking execution.",
                        CORAL,
                    ),
                    spacing="3",
                    align="start",
                    width="100%",
                ),
                columns="2",
                spacing="6",
                width="100%",
            ),
            background="#FFF8F6",
            border_left=f"4px solid {CORAL}",
            padding="1.25rem",
            width="100%",
        ),
        _limitations_section(
            "05 \u00b7 CONTROLS AND FUTURE WORK",
            "What is controlled now and what still requires evidence",
            "Current safeguards remain separate from proposed extensions.",
        ),
        rx.grid(
            rx.vstack(
                rx.text(
                    "CURRENT CONTROLS",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                _plain_limit_item(
                    "check",
                    "Standard-fare population and prior-only feature contract.",
                    GREEN,
                ),
                _plain_limit_item(
                    "check",
                    "Observed-and-mature target scoring.",
                    GREEN,
                ),
                _plain_limit_item(
                    "check",
                    "Explicit canonical, off-grid, and unsupported output modes.",
                    GREEN,
                ),
                _plain_limit_item(
                    "check",
                    "Frozen threshold, model version, data cutoff, and as-of time.",
                    GREEN,
                ),
                spacing="3",
                align="start",
                width="100%",
                padding_right="2rem",
            ),
            rx.vstack(
                rx.text(
                    "FUTURE EVIDENCE",
                    size="1",
                    color=BLUE,
                    weight="bold",
                ),
                _plain_limit_item(
                    "arrow-right",
                    "Longer post-deployment drift monitoring and scheduled refits.",
                    BLUE,
                ),
                _plain_limit_item(
                    "arrow-right",
                    "Wider route-airline coverage under the same validation protocol.",
                    BLUE,
                ),
                _plain_limit_item(
                    "arrow-right",
                    "Lawful inventory, demand, and promotion proxies when available.",
                    BLUE,
                ),
                _plain_limit_item(
                    "arrow-right",
                    "Daily labels before any off-grid classification policy.",
                    BLUE,
                ),
                spacing="3",
                align="start",
                width="100%",
                padding_left="2rem",
                border_left=f"1px solid {LINE}",
            ),
            columns="2",
            spacing="0",
            width="100%",
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            padding="1.25rem",
        ),
        rx.text(
            "Reporting basis: Model Cards for Model Reporting; Datasheets for "
            "Datasets; NIST AI Risk Management Framework 1.0.",
            size="1",
            color=MUTED,
            font_style="italic",
        ),
        spacing="6",
        width="100%",
    )


def reserved_page(title: rx.Var) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(
                "SKYFARE ANALYSIS",
                size="1",
                color=TEAL,
                weight="bold",
            ),
            rx.heading(
                title,
                size="7",
                color=INK,
                letter_spacing="0",
            ),
            rx.text(
                "This section is reserved for the next reviewed stage.",
                size="2",
                color=MUTED,
            ),
            spacing="2",
            align="start",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="2rem",
        min_height="360px",
        width="100%",
    )


def query_panel() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.text(
                    "FLIGHT SEARCH",
                    size="1",
                    color=TEAL,
                    weight="bold",
                ),
                rx.heading(
                    "Compare future departure fares",
                    size="6",
                    color=INK,
                    letter_spacing="0",
                ),
                rx.text(
                    "Historical replay \u00b7 standard-fare population",
                    size="2",
                    color=MUTED,
                ),
                spacing="1",
                align="start",
            ),
            rx.spacer(),
            rx.vstack(
                rx.text("DATA CUTOFF", size="1", color=MUTED, weight="bold"),
                rx.text(
                    DemoState.cutoff,
                    size="2",
                    color=INK,
                    font_family=MONO,
                    weight="bold",
                ),
                spacing="1",
                align="end",
            ),
            width="100%",
            align="start",
        ),
        rx.grid(
            field(
                "BOOKING DATE",
                calendar_control(dynamic_date_select()),
            ),
            field(
                "TIME PERIOD",
                rx.segmented_control.root(
                    rx.segmented_control.item("Morning \u00b7 AM", value="AM"),
                    rx.segmented_control.item("Evening \u00b7 PM", value="PM"),
                    value=DemoState.session_label,
                    on_change=DemoState.set_session_label,
                    width="100%",
                    class_name="time-period-toggle",
                ),
            ),
            field(
                "FROM",
                linked_select_field(
                    DemoState.origin,
                    DemoState.set_origin,
                    "Origin",
                    DemoState.origin_options,
                ),
            ),
            field(
                "TO",
                linked_select_field(
                    DemoState.destination,
                    DemoState.set_destination,
                    "Destination",
                    DemoState.destination_options,
                ),
            ),
            field(
                "FLIGHT DATE",
                calendar_control(dynamic_flight_date_select()),
            ),
            field(
                "AIRLINE",
                linked_select_field(
                    rx.cond(
                        DemoState.airline == "",
                        "ALL",
                        DemoState.airline,
                    ),
                    DemoState.set_airline,
                    "All airlines",
                    DemoState.airline_options,
                ),
            ),
            columns=rx.breakpoints(initial="1", sm="2", lg="3"),
            spacing="4",
            width="100%",
            margin_top="1.5rem",
        ),
        rx.center(
            rx.button(
                rx.cond(
                    DemoState.searching,
                    rx.hstack(
                        rx.spinner(size="2"),
                        rx.text("Running ensemble"),
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.icon("search", size=18),
                        rx.text("Find"),
                        spacing="2",
                    ),
                ),
                on_click=DemoState.run_search,
                disabled=~DemoState.can_search,
                background=TEAL,
                color="white",
                height="44px",
                padding_x="1.25rem",
                cursor="pointer",
            ),
            width="100%",
            margin_top="1.25rem",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="8px",
        padding="1.5rem",
        width="100%",
    )


def airline_logo(path: rx.Var, alt: rx.Var) -> rx.Component:
    return rx.box(
        rx.image(
            src=path,
            alt=alt,
            width="46px",
            height="46px",
            object_fit="contain",
        ),
        width="48px",
        height="48px",
        display="flex",
        align_items="center",
        justify_content="center",
        flex_shrink="0",
    )


def prediction_summary(item: rx.Var, compact: bool = False) -> rx.Component:
    price_block = rx.vstack(
        rx.hstack(
            rx.text(
                item["price"],
                size="2" if compact else "4",
                color=INK,
                weight="bold",
                font_family=MONO,
                white_space="nowrap",
            ),
            rx.text("VND", size="1", color=MUTED),
            spacing="1",
            align="baseline",
        ),
        rx.text(
            item["anchor_delta"],
            size="1",
            color=rx.cond(item["anchor_down"], GREEN, CORAL),
        ),
        spacing="0",
        align="end" if not compact else "start",
    )
    action_block = rx.cond(
        item["has_action"],
        rx.vstack(
            rx.badge(
                item["action"],
                color_scheme=rx.cond(
                    item["is_wait"], "orange", "green"
                ),
                variant="solid",
                size="2",
                font_size="0.9rem",
                min_width="58px",
                padding="0.38rem 0.65rem",
                justify_content="center",
            ),
            rx.text(
                item["drop_probability"],
                size="1",
                color=MUTED,
                font_family=MONO,
            ),
            spacing="1",
            align="center",
        ),
        rx.vstack(
            rx.badge(
                "FARE ONLY",
                color_scheme="gray",
                variant="soft",
            ),
            rx.text(item["action_note"], size="1", color=MUTED),
            spacing="1",
            align="center",
        ),
    )
    if compact:
        return rx.hstack(
            price_block,
            action_block,
            spacing="3",
            align="center",
        )
    return rx.grid(
        price_block,
        action_block,
        grid_template_columns="180px 72px",
        column_gap="1.5rem",
        align_items="center",
        justify_content="center",
        width="100%",
    )


def result_row(item: rx.Var) -> rx.Component:
    expanded = DemoState.expanded_slot == item["slot_id"]
    return rx.box(
        rx.grid(
            rx.text(
                item["rank"],
                size="2",
                color=MUTED,
                text_align="center",
                font_family=MONO,
            ),
            airline_logo(item["logo_path"], item["logo_alt"]),
            rx.vstack(
                rx.hstack(
                    rx.text(
                        item["airline_name"],
                        size="2",
                        color=INK,
                        weight="bold",
                    ),
                    rx.cond(
                        item["is_cheapest"],
                        rx.badge(
                            "BEST FARE",
                            color_scheme="teal",
                            variant="soft",
                        ),
                    ),
                    spacing="2",
                ),
                spacing="0",
                align="start",
                width="100%",
            ),
            rx.vstack(
                rx.text(
                    item["departure_time"],
                    size="5",
                    color=INK,
                    weight="bold",
                    font_family=MONO,
                ),
                spacing="0",
                align="center",
                width="100%",
            ),
            rx.box(
                prediction_summary(item),
                grid_column="5 / 7",
                grid_row="1",
                width="100%",
                display="flex",
                justify_content="center",
            ),
            rx.icon_button(
                rx.icon(
                    rx.cond(expanded, "chevron-up", "chevron-down"),
                    size=17,
                ),
                variant="ghost",
                color_scheme="gray",
                on_click=lambda: DemoState.toggle_slot(item["slot_id"]),
                cursor="pointer",
                aria_label="Toggle flight details",
                grid_column="6",
                grid_row="1",
                justify_self="end",
                z_index="1",
            ),
            grid_template_columns=(
                "22px 50px minmax(90px, 1fr) "
                "80px minmax(155px, 1.25fr) 24px"
            ),
            class_name="flight-row-wide",
            width="100%",
            align="center",
            spacing="2",
            padding="0.85rem 0.75rem",
        ),
        rx.vstack(
            rx.hstack(
                rx.text(
                    item["rank"],
                    size="1",
                    color=MUTED,
                    width="20px",
                    text_align="center",
                    font_family=MONO,
                    flex_shrink="0",
                ),
                airline_logo(item["logo_path"], item["logo_alt"]),
                rx.vstack(
                    rx.hstack(
                        rx.text(
                            item["airline_name"],
                            size="2",
                            color=INK,
                            weight="bold",
                        ),
                        rx.cond(
                            item["is_cheapest"],
                            rx.badge(
                                "BEST",
                                color_scheme="teal",
                                variant="soft",
                            ),
                        ),
                        spacing="2",
                    ),
                    rx.text(
                        item["departure_time"],
                        size="4",
                        color=INK,
                        weight="bold",
                        font_family=MONO,
                    ),
                    spacing="0",
                    align="start",
                    flex="1",
                    min_width="0",
                ),
                rx.spacer(),
                rx.icon_button(
                    rx.icon(
                        rx.cond(expanded, "chevron-up", "chevron-down"),
                        size=17,
                    ),
                    variant="ghost",
                    color_scheme="gray",
                    on_click=lambda: DemoState.toggle_slot(item["slot_id"]),
                    cursor="pointer",
                    aria_label="Toggle flight details",
                    flex_shrink="0",
                ),
                width="100%",
                align="center",
                spacing="2",
            ),
            rx.box(
                prediction_summary(item),
                margin_left="82px",
            ),
            width="100%",
            align="start",
            spacing="2",
            padding="0.8rem 0.7rem",
            class_name="flight-row-mobile",
        ),
        rx.cond(
            expanded,
            rx.box(
                rx.grid(
                    detail_tile(
                        "brain-circuit",
                        "Fare estimation models",
                        item["fare_models"],
                        "Frozen ensemble weights; CatBoost handles fallback cases.",
                    ),
                    detail_tile(
                        "chart-no-axes-combined",
                        "Price-drop models",
                        item["exact_models"],
                        "Used for DROP5 probability only at observed booking windows.",
                    ),
                    columns=rx.breakpoints(initial="1", lg="2"),
                    spacing="3",
                    width="100%",
                ),
                rx.grid(
                    detail_tile(
                        "badge-dollar-sign",
                        "Reference fare",
                        item["baseline_display"],
                        "Strictly-prior reference; no future fare is used.",
                    ),
                    detail_tile(
                        "database",
                        "Evidence level",
                        item["regime_label"],
                        item["regime_help"],
                    ),
                    detail_tile(
                        "history",
                        "Schedule history",
                        item["history_label"],
                        item["history_help"],
                    ),
                    detail_tile(
                        "layers-3",
                        "Reference support",
                        item["anchor_label"],
                        item["anchor_help"],
                    ),
                    detail_tile(
                        "calendar-check-2",
                        "Prediction coverage",
                        item["support_label"],
                        item["support_help"],
                    ),
                    columns=rx.breakpoints(initial="1", sm="2", lg="3"),
                    spacing="3",
                    width="100%",
                    margin_top="0.75rem",
                ),
                rx.cond(
                    item["interpolation_interval"] != "",
                    rx.callout(
                        item["interpolation_note"],
                        icon="between-horizontal-start",
                        color_scheme="gray",
                        size="1",
                        margin_top="0.8rem",
                    ),
                ),
                background="#F5F9F8",
                border_top=f"1px solid {LINE}",
                padding="1.1rem 1.25rem",
            ),
        ),
        border_bottom=f"1px solid {LINE}",
        border_left=rx.cond(
            item["is_cheapest"],
            f"4px solid {TEAL}",
            "4px solid transparent",
        ),
        background=rx.cond(item["is_cheapest"], "#F1FAF7", PANEL),
        width="100%",
    )


def detail_tile(
    icon: str,
    label: str,
    value: rx.Var,
    help_text: rx.Var | str,
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=16, color=TEAL),
                rx.text(label, size="1", color=MUTED, weight="bold"),
                spacing="2",
                align="center",
            ),
            rx.text(
                value,
                size="2",
                color=INK,
                weight="bold",
                word_break="break-word",
            ),
            rx.text(
                help_text,
                size="1",
                color=MUTED,
                line_height="1.45",
            ),
            spacing="2",
            align="start",
        ),
        background=PANEL,
        border=f"1px solid {LINE}",
        border_radius="6px",
        padding="0.9rem",
        min_height="116px",
        width="100%",
    )


def results_panel() -> rx.Component:
    return rx.cond(
        DemoState.results.length() > 0,
        rx.box(
            rx.box(
                rx.grid(
                    rx.hstack(
                        airline_logo(
                            DemoState.cheapest_logo,
                            DemoState.cheapest_logo_alt,
                        ),
                        rx.vstack(
                            rx.text(
                                "BEST FARE FOUND",
                                size="1",
                                color=TEAL,
                                weight="bold",
                            ),
                            rx.hstack(
                                rx.text(
                                    DemoState.cheapest_price,
                                    size="6",
                                    color=INK,
                                    weight="bold",
                                    font_family=MONO,
                                    white_space="nowrap",
                                ),
                                rx.text("VND", size="2", color=MUTED),
                                spacing="2",
                                align="baseline",
                            ),
                            spacing="1",
                            align="start",
                        ),
                        spacing="3",
                        align="center",
                        grid_column="2 / 4",
                    ),
                    rx.vstack(
                        rx.text(
                            "DEPARTURE",
                            size="1",
                            color=MUTED,
                            weight="bold",
                        ),
                        rx.text(
                            DemoState.cheapest_context,
                            size="5",
                            color=INK,
                            weight="bold",
                            font_family=MONO,
                        ),
                        spacing="1",
                        align="center",
                        grid_column="4",
                        width="100%",
                    ),
                    rx.hstack(
                        metric_chip(
                            "Flights", DemoState.result_count.to_string()
                        ),
                        metric_chip(
                            "Days ahead", DemoState.dud.to_string()
                        ),
                        metric_chip("Processing", DemoState.query_time),
                        spacing="2",
                        align="center",
                        justify="center",
                        flex_wrap="nowrap",
                        grid_column="5 / 7",
                    ),
                    grid_template_columns=(
                        "22px 50px minmax(90px, 1fr) "
                        "80px minmax(155px, 1.25fr) 24px"
                    ),
                    width="100%",
                    align="center",
                    spacing="2",
                    class_name="best-fare-summary",
                ),
                rx.cond(
                    DemoState.airline != "",
                    rx.center(
                        rx.badge(
                            "Cheapest first \u00b7 remaining flights ordered by time",
                            color_scheme="teal",
                            variant="soft",
                        ),
                        width="100%",
                        margin_top="0.9rem",
                    ),
                ),
                padding="1.25rem 0.75rem",
                background=TEAL_SOFT,
            ),
            rx.foreach(
                DemoState.results,
                lambda item: rx.cond(
                    item["is_cheapest"],
                    rx.fragment(),
                    result_row(item),
                ),
            ),
            background=PANEL,
            border=f"1px solid {LINE}",
            border_radius="8px",
            overflow="hidden",
            width="100%",
        ),
    )


def metric_chip(label: str, value: rx.Var) -> rx.Component:
    return rx.vstack(
        rx.text(label, size="1", color=MUTED),
        rx.text(
            value,
            size="2",
            color=INK,
            weight="bold",
            font_family=MONO,
        ),
        spacing="0",
        align="center",
        min_width="76px",
        padding="0.55rem 0.7rem",
        background="#F3F7F6",
        border=f"1px solid {LINE}",
        border_radius="6px",
    )


def status_footer() -> rx.Component:
    return rx.hstack(
        rx.text(
            DemoState.model_version,
            size="1",
            color=MUTED,
            font_family=MONO,
        ),
        rx.spacer(),
        rx.text(
            "Model preload " + DemoState.load_time,
            size="1",
            color=MUTED,
        ),
        width="100%",
        padding_y="1rem",
    )


def demo_page() -> rx.Component:
    return rx.vstack(
        query_panel(),
        rx.cond(
            DemoState.error != "",
            rx.callout(
                DemoState.error,
                icon="triangle-alert",
                color_scheme="red",
                width="100%",
            ),
        ),
        rx.cond(
            DemoState.searching,
            rx.box(
                rx.vstack(
                    rx.spinner(size="3", color=TEAL),
                    rx.text(
                        "Running frozen ensembles",
                        size="2",
                        color=INK,
                        weight="bold",
                    ),
                    rx.text(
                        "Fare ranking and DROP5 policy",
                        size="1",
                        color=MUTED,
                    ),
                    spacing="2",
                    align="center",
                ),
                width="100%",
                min_height="240px",
                display="flex",
                align_items="center",
                justify_content="center",
            ),
            results_panel(),
        ),
        status_footer(),
        spacing="4",
        width="100%",
    )


def page_content() -> rx.Component:
    return rx.cond(
        DemoState.active_tab == "Overview",
        overview_page(),
        rx.cond(
            DemoState.active_tab == "Data Analysis",
            data_analysis_page(),
            rx.cond(
                DemoState.active_tab == "Feature Engineering",
                feature_engineering_page(),
                rx.cond(
                    DemoState.active_tab == "Modelling",
                    modelling_page(),
                    rx.cond(
                        DemoState.active_tab == "Evaluation",
                        evaluation_page(),
                        rx.cond(
                            DemoState.active_tab == "Limitations",
                            limitations_page(),
                            rx.cond(
                                DemoState.active_tab == "Interactive Demo",
                                demo_page(),
                                reserved_page(DemoState.active_tab),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def index() -> rx.Component:
    return rx.box(
        model_header(),
        rx.box(
            page_content(),
            max_width="1240px",
            width="100%",
            margin="0 auto",
            padding=rx.breakpoints(
                initial="1rem", md="1.5rem", lg="2rem"
            ),
        ),
        background=PAGE,
        min_height="100vh",
        width="100%",
        color=INK,
    )


app = rx.App(
    style={
        "font_family": "Inter, ui-sans-serif, system-ui, sans-serif",
        "letter_spacing": "0",
    },
    stylesheets=["/styles.css"],
)
app.add_page(
    index,
    route="/",
    title="SkyFare \u00b7 Fare Intelligence",
    on_load=DemoState.initialize,
)
