#!/bin/bash

# start_jupyter_gpu.sh
# Description: Activates the Conda environment and starts Jupyter Lab on the
# compute node, configured to listen on all IPs (0.0.0.0) so it can be
# accessed via a remote SSH tunnel.

# --- Configuration ---
ENV_PATH="$HOME/miniconda3/envs/speed_camara_enforcement"
JUPYTER_PORT="9999"

# --- Script Logic ---

# 1. Check if the Conda environment path exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Conda environment not found at $ENV_PATH" >&2
    echo "Please update the ENV_PATH variable inside this script." >&2
    exit 1
fi

echo "--- Activating Conda Environment ---"
# Esta línea carga Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
# Activar el entorno
conda activate "$ENV_PATH"
echo "Active environment: $CONDA_DEFAULT_ENV"

echo ""
echo "--- Starting Jupyter Lab ---"

echo ""
echo "--- Important Instructions ---"
echo "1. Keep this terminal window open; closing it stops the server."
echo "2. Note the 'token' in the URL printed above."
echo "3. From your LOCAL PC, run the SSH tunnel command:"
echo "   ssh -L $JUPYTER_PORT:$(hostname -f):$JUPYTER_PORT your_username@login_node_ip"
echo "4. Access the notebook in your local browser at: http://localhost:$JUPYTER_PORT/lab?token=..."

# 2. Run the Jupyter Lab command
# --no-browser prevents the remote server from trying to open a GUI browser.
# --port sets the port for the tunnel.
# --ip=0.0.0.0 is CRITICAL for accepting the forwarded connection from the login node.
jupyter lab --no-browser --port="$JUPYTER_PORT" --ip=0.0.0.0
