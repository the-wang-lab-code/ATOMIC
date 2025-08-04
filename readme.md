# ATOMIC: Autonomous Technology for Optical Microscopy \& Intelligent Characterization

## Overview

This project is an advanced automated system that uses a Large Language Model (LLM) as an intelligent agent to autonomously control an optical microscope (OM) and perform image analysis. The system's primary goal is to automate repetitive tasks in materials science, such as locating, identifying, and analyzing 2D material flakes on a sample.

Users can provide a high-level final objective via an API. The system will then automatically plan and execute a series of actions, including moving the sample stage, auto-focusing, capturing images, and using advanced image segmentation and classification algorithms for real-time analysis to accomplish the specified task.

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

## Core Features

* **Intelligent Agent Control**: Integrates the OpenAI GPT model as an intelligent agent to understand high-level user commands and make autonomous decisions.
* **Automated Microscope Operation**: Enables precise control over the microscope's stage movement, auto-focus, and auto-exposure through an interface with ZEISS ZEN software.
* **Advanced Image Analysis**:
    * **SAM (Segment Anything Model)**: Utilizes SAM for high-quality image segmentation to accurately identify different regions in the image.
    * **K-means & GPT-powered Classification**: An innovative K-means clustering workflow that leverages a GPT model to intelligently determine the optimal number of clusters and classify the results (e.g., substrate, impurities, 2D materials).
* **Web Service Interface**: Provides an HTTP interface via Flask, making it easy for users or other programs to integrate and interact with the system.

## Installation and Setup

1.  **Prerequisites**:
    * A running instance of ZEISS ZEN microscope control software.
    * Python 3.x environment.

2.  **Install Dependencies**:
    Clone the repository and install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**:
    Open `control/utils/sensitives.py` and enter your OpenAI API key:
    ```python
    # control/utils/sensitives.py
    OPENAI_API_KEY = 'your-api-key' # Replace with your key
    ```

## Usage

1.  **Start the Main Service**:
    Run the `service_om.py` and `interface_autonomous_om.py` script to start the Flask server on port `8050`.
    ```bash
    python control/service_om.py
    python control/interface_autonomous_om.py
    ```

2.  **Send Task Command**:
    Use a script or any HTTP client to send a POST request to the server to start a task, as demonstrated in `control/interactive.ipynb`.

    **Example**:
    ```python
    import requests

    url = '[http://127.0.0.1:8050/om_imaging](http://127.0.0.1:8050/om_imaging)'
    data = {
        "final_objective": "Scan a 3000x3000 micrometer area, find all 2D material flakes larger than 200 micrometers, and move to the center of the largest one."
    }
    response = requests.post(url, json=data)

    print(response.status_code)
    print(response.text)
    ```
