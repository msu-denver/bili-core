"""
Module: app

This module serves as the main entry point for the Streamlit application. It
includes functions to configure the Streamlit environment, determine the
appropriate checkpointer, and execute the main application logic.

Functions:
    - main(): Sets up the main entry point for the Streamlit application,
              configures the Streamlit page layout, determines the appropriate
              checkpointer, and runs the main application page logic.
    - configure_streamlit(): Configures the Streamlit application for the Bili
                             Core Sandbox, setting the page configurations,
                             displaying the application logo, and providing
                             initial instructions for the user interface.

Dependencies:
    - os: Provides a way of using operating system dependent functionality.
    - pathlib: Provides object-oriented filesystem paths.
    - streamlit: Streamlit library for building web applications.
    - PIL: Python Imaging Library for image processing.
    - bili.checkpointers.checkpointer_functions: Imports get_checkpointer
      function to determine the appropriate checkpointer (PostgresSaver or
      MemorySaver).
    - bili.streamlit.ui.auth_ui: Imports check_auth and initialize_auth_manager
      functions for authentication.
    - bili.streamlit.ui.chat_interface: Imports run_app_page function to run
      the main application logic.

Usage:
    This module is intended to be executed as a script to start the Streamlit
    application. It configures the Streamlit environment, determines the
    appropriate checkpointer, and runs the main application page logic.

Example:
    To run the Streamlit application, execute the following command:

    ```bash
    python -m bili.app
    ```

    This will start the Streamlit application with the configured settings and
    checkpointer.
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from bili.checkpointers.checkpointer_functions import get_checkpointer
from bili.streamlit_ui.ui.auth_ui import check_auth, initialize_auth_manager
from bili.streamlit_ui.ui.chat_interface import run_app_page

# Disable tokenizers parallelism to avoid issues with Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    Main entry point for the Streamlit application.

    This function performs the following steps:
    1. Configures the Streamlit application,
       including page title, logo, and introduction text.
    2. Initializes the authentication manager with specified providers.
    3. Sets up Streamlit authentication to verify if the user is authenticated.
    4. Determines the appropriate checkpointer for application state management
       (PostgresSaver or MemorySaver).
    5. Executes the main application page logic.

    :return: None
    """
    # Set the Streamlit page title, logo, and introduction
    configure_streamlit()

    # Initialize the appropriate AuthManager
    st.session_state.auth_manager = initialize_auth_manager(
        auth_provider_name="sqlite",
        profile_provider_name="sqlite",
        role_provider_name="sqlite",
    )

    # Setup Streamlit authentication and check if the user is authenticated
    check_auth()

    # Get the appropriate checkpointer (PostgresSaver or MemorySaver)
    checkpointer = get_checkpointer()

    # Run the main Streamlit application page logic
    run_app_page(checkpointer)


def configure_streamlit():
    """
    Configure the Streamlit application by setting up its layout, appearance, and initial
    content. This includes loading and displaying a logo image, configuring the page title,
    icon, and layout, and providing an initial user interface with explanatory text and
    instructions for interacting with the application.

    :return: None
    """
    logo_path = Path(__file__).parent.joinpath("images", "logo.png").as_posix()
    logo = Image.open(logo_path)
    st.set_page_config(
        page_title="Bili Core Sandbox Application", page_icon=logo, layout="wide"
    )
    st.image(logo, width=100, use_container_width=False)

    st.markdown(
        """<h1>Welcome to the BiliCore LLM Comparison Sandbox!</h1>
        <h2><a href="https://github.com/msu-denver/bili-core">BiliCore on GitHub</a></h2>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p><h3>This sandbox allows you to test and configure various
        aspects of the chatbot system, including Language Models (LLMs), tools, and prompts.
        The configuration panels below let you customize the chatbot's behavior.</h3></p>
        <p><h3>After making changes, click "Load Configuration" to apply them.</h3></p>
        <p><h3><strong>Please note:</strong> If you refresh the page, your session will restart,
        and you will lose any unsaved changes.</h3></p>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")


# Run the main function when the script is executed.
if __name__ == "__main__":
    main()
