# Drift Detection Monitoring with Prometheus & Grafana

This project provides a simple, containerized monitoring stack for a Drift Detection pipeline using Docker, Python (Flask), Prometheus, and Grafana.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

## How to Run

1.  **Start the entire stack:**
    Open your terminal in the `Module6 monitoring` directory and run the following command. This will take some time the first time you run it as it needs to download the base Python image and install all the libraries (including PyTorch).
    ```bash
    docker-compose up --build -d
    ```

2.  **Access Grafana:**
    Open your web browser and navigate to `http://localhost:3000`.
    You should see the new "Drift Detection Monitoring" dashboard.

3.  **Run the Drift Detection Pipeline:**
    To see the dashboard update, you need to trigger the drift detection pipeline. Open a new terminal and run this command:

    ```bash
    curl http://localhost:5001/run
    ```
    This will start the full training and detection process inside the container. It will take a few minutes to complete. You will see progress updates in your Docker logs. When it finishes, the panels on your Grafana dashboard will update with the final results.

4.  **Access Prometheus (Optional):**
    You can also view the raw metrics and service status directly in Prometheus by navigating to `http://localhost:9090`. Go to the "Status" -> "Targets" page to confirm that Prometheus is successfully scraping the `app`.

## How to Stop

To stop and remove all the containers, run the following command in the `Module6 monitoring` directory:
```bash
docker-compose down
``` 