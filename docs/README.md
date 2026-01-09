# Robomesh Integration

## Installation

First, install the tomwebrtc component.

> **Note**: This is a placeholder as the `tomwebrtc` repository is not public yet.

## Usage

1. Start the webrtc server:

   ```bash
   cd tomwebrtc
   go run main.go
   ```

2. Run the current server:

   ```bash
   python scripts/robomesh_server.py
   ```

3. Interaction:

   In the webpage, send "random dance", and you will see the robot dance in the pybullet and on robomesh webpage.
