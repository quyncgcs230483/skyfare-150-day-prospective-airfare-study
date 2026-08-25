import reflex as rx

config = rx.Config(
    app_name="skyfare_inference_demo",
    plugins=[
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="light",
                accent_color="teal",
                gray_color="slate",
                radius="medium",
            ),
        ),
    ],
)
