# CARLA Simulator Workspace Structure

This document provides an overview of the folder structure and contents in the CARLA 0.9.9.4 workspace. CARLA is an open-source simulator for autonomous driving research, built on Unreal Engine.

## Root Directory Structure

### WindowsNoEditor/
This folder contains the Windows build of CARLA without the Unreal Engine Editor. It includes:
- **CHANGELOG**: Version history and release notes
- **Dockerfile**: Docker configuration for containerized deployment
- **LICENSE**: Licensing information
- **README**: Basic setup and usage instructions
- **CarlaUE4/**: The main Unreal Engine project directory

### CarlaUE4/
The core Unreal Engine 4 project for CARLA. Contains:
- **CarlaUE4.uproject**: Unreal Engine project file
- **Binaries/Win64/**: Compiled binaries for Windows
- **Config/**: Engine and game configuration files (DefaultEngine.ini, DefaultGame.ini, etc.)
- **Content/Carla/**: Game assets, blueprints, and CARLA-specific content
- **Plugins/Carla/**: CARLA-specific Unreal Engine plugins

### Co-Simulation/
Integration modules for co-simulation with other traffic simulators:
- **PTV-Vissim/**: Integration with PTV Vissim traffic simulator
  - `run_synchronization.py`: Main synchronization script
  - `data/`: Data files for Vissim integration
  - `examples/`: Example scripts and configurations
  - `vissim_integration/`: Integration modules
- **Sumo/**: Integration with SUMO (Simulation of Urban MObility)
  - `requirements.txt`: Python dependencies
  - `run_synchronization.py`: Synchronization script
  - `spawn_npc_sumo.py`: NPC spawning script
  - `data/`: SUMO network and route data
  - `examples/`: Example configurations
  - `sumo_integration/`: Core integration code
  - `util/`: Utility functions

### Engine/
Unreal Engine 4 installation and assets:
- **Binaries/ThirdParty/**: Third-party binaries and libraries
- **Config/**: Base engine configuration files
- **Content/**: Engine content including materials, meshes, sounds, and editor resources
- **Plugins/**: Engine plugins for various features (AI, Media, MovieScene, etc.)

### HDMaps/
High-definition map data and documentation:
- **README**: Information about HD map usage and formats

### PythonAPI/
Python client API for interacting with CARLA:
- **carla/**: Main Python module
  - `requirements.txt`: Python dependencies
  - `scene_layout.py`: Scene layout utilities
  - `agents/`: Autonomous agent implementations
- **examples/**: Python scripts demonstrating CARLA usage
  - Various example scripts for manual control, automatic control, weather simulation, etc.
- **util/**: Utility scripts and helper functions

## Usage Notes

- **WindowsNoEditor/** is the main entry point for running CARLA on Windows
- **PythonAPI/** contains the Python client for scripting and automation
- **Co-Simulation/** folders enable integration with external traffic simulators
- **CarlaUE4/** requires Unreal Engine 4 to modify or rebuild
- **Engine/** contains the Unreal Engine installation (typically not modified)

For detailed setup instructions, refer to the README files in each major directory.