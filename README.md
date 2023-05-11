# CLIP-BOSS

The pipeline & demo runs based off of two python scripts:
 - `fetch_only_pickup.py` feeds input to the compute server, gets back classification results and performs action
 - `network_compute_server.py` takes in input feeds from spot and runs the segmentation & CLIP model. It listens for requests from Spot.

The demo runs with the following command:

```
python network_compute_server.py <spot_ip>
python fetch_only_pickup.py -s fetch-server -m clip-classifier <spot_ip>
```