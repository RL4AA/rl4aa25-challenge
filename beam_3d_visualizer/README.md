# ARES 3D Visualization

This repository contains a Python-based simulation control system and a JavaScript-based 3D visualization application for a particle beam lattice. Follow the instructions below to set up the environment and dependencies.

## Prerequisites

Ensure you have the following installed on your system:
- [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- [Node.js & npm](https://nodejs.org/en/download/)

## Setting up the Environment

### 1. Create and Activate the Conda Environment
```bash
conda create --name rlaa2025 python=3.11 -y
conda activate rlaa2025
```

### 2. Install Python Dependencies
```bash
pip install -r simulation_controller/requirements.txt
python -m pip install git+https://github.com/ocelot-collab/ocelot.git@v22.12.0
pip install git+https://github.com/chrisjcc/cheetah.git@feature/3d_lattice_viewer
```

### 3. Set Up JavaScript Dependencies
Navigate to the JavaScript project directory (e.g., `beam_3d_visualizer/`) and install the necessary dependencies:
```bash
cd beam_3d_visualizer  # Adjust the path as necessary
npm install
npm audit fix
npm install plotly.js-dist@3.0.1
npm install -g vite@6.2.0
```

### 4. Run the Application
Start the development server:
```bash
npm run start
```

## Additional Notes
- Ensure the Python simulation control system is running before launching the visualization.
- If encountering issues with dependencies, try reinstalling them or using a clean environment.
- For further troubleshooting, check the respective documentation of `npm`, `pip`, or `conda`.

## License
[Specify your project's license here]

## Contact
For issues or contributions, feel free to open a GitHub issue or reach out to the maintainers.
