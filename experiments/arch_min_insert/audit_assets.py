"""Read authored CAD and collision properties with the installed USD runtime."""
import argparse,json
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
import numpy as np
root=Path('../../taskboard/fmb_example')
root=Path('../taskboard/fmb_example')
results={}
for filename in ['Medium_Board_DarkBlue_Updated.usd','Medium_Short_Hexagon_Green_Updated.usd']:
    stage=Usd.Stage.Open(str(root/filename));cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    rows=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            row={'path':str(prim.GetPath()),'type':prim.GetTypeName(),'schemas':list(prim.GetAppliedSchemas()),'bounds_min':list(bounds.GetMin()),'bounds_max':list(bounds.GetMax()),'attrs':{attr.GetName():str(attr.Get()) for attr in prim.GetAttributes() if any(k in attr.GetName().lower() for k in ['physics','approximation'])}}
            if prim.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(prim);x=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                pts=np.array([list(x.Transform(v)) for v in mesh.GetPointsAttr().Get()]);row['vertices']=len(pts)
                np.savez_compressed(a.output.parent/(filename+'.mesh.npz'),points=pts,counts=np.array(mesh.GetFaceVertexCountsAttr().Get()),indices=np.array(mesh.GetFaceVertexIndicesAttr().Get()))
            rows.append(row)
    results[filename]={'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'prims':rows}
a.output.write_text(json.dumps(results,indent=2)+'\n');print('CAD_AUDIT_COMPLETE',a.output);app.close()
