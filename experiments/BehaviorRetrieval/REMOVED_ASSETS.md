# Removed Assets for GitHub Compatibility

To keep the repository under GitHub's size limits, the following large binary assets have been removed:

## Removed Files (187MB → 22MB)

### ShapeNet 3D Models
- `roboverse/roboverse/assets/bullet-objects/ShapeNetCore/` - Complete ShapeNet dataset
  - Contains thousands of 3D object models (.obj files)
  - Used for robotic simulation environments
  - **Size**: ~180MB

### Large Mesh Files
- `roboverse/roboverse/assets/room_descriptions/meshes/` - 3D mesh objects
- `roboverse/roboverse/assets/room_descriptions/lamp/lampe.obj` - Large 3D model
  - **Size**: ~5MB

## How to Restore (If Needed)

If you need these assets for experiments:

1. **Download ShapeNet Core.v2**:
   ```bash
   # Register at https://www.shapenet.org/
   # Download ShapeNet Core.v2 dataset
   # Extract to: experiments/BehaviorRetrieval/roboverse/roboverse/assets/bullet-objects/
   ```

2. **Alternative: Use Simplified Models**:
   ```bash
   # Use basic geometric shapes instead of complex ShapeNet models
   # Modify roboverse configs to use built-in primitive shapes
   ```

## Impact on Functionality

- **BehaviorRetrieval experiments**: May need simpler object models
- **Roboverse simulations**: Will fall back to basic geometric shapes
- **Core RT-Cache functionality**: **Not affected** - these are only for baseline experiments

## Note

The core RT-Cache system (embedding generation, retrieval, FRANKA integration) does not depend on these 3D models. They were only used for robotic simulation baselines.