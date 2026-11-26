"""Run in 3D Slicer: Slicer --python-script build_demo_scene.py  -> similarity_demo.mrb (data are LPS mm -> RAS)."""
import slicer, vtk
O = "/Volumes/Extreme SSD/2d_3d_regis dataset/New_version/Data/outputs/predict_demo/similarity/slicer"
items = [("similarity_input_skin.vtp", "Input skin (similarity.vtp)", (0.8, 0.8, 0.8), 0.25, True),
         ("similarity_registered_skin.vtp", "Registered FE skin", (1.0, 0.6, 0.1), 0.3, False),
         ("similarity_pred_skull.vtp", "Predicted skull", (0.95, 0.95, 0.85), 0.45, True),
         ("similarity_pred_brain.vtp", "Predicted brain", (0.85, 0.5, 0.5), 0.9, True),
         ("similarity_pred_mandible.vtp", "Predicted mandible", (0.5, 0.7, 0.9), 0.9, True),
         ("similarity_FE_head_untangled.vtu", "FE head (sigma_mm)", (1, 1, 1), 1.0, False)]
flip = vtk.vtkMatrix4x4(); flip.SetElement(0, 0, -1); flip.SetElement(1, 1, -1)
tr = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "LPS_to_RAS"); tr.SetMatrixTransformToParent(flip)
for f, name, rgb, op, vis in items:
    n = slicer.util.loadNodeFromFile(f"{O}/{f}", "ModelFile", {"coordinateSystem": "RAS"}); n.SetName(name); n.SetAndObserveTransformNodeID(tr.GetID())
    d = n.GetDisplayNode(); d.SetColor(*rgb); d.SetOpacity(op); d.SetVisibility(vis)
    if f.endswith(".vtu"):
        n.GetMesh().GetPointData().SetActiveScalars("sigma_mm"); d.SetScalarVisibility(True); d.SetActiveScalarName("sigma_mm")
        d.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileViridis.txt"); d.SetScalarRangeFlag(0); d.SetScalarRange(0, 8); d.SetSliceIntersectionVisibility(True)
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView); slicer.util.resetThreeDViews()
print("scene ready")
slicer.app.processEvents()
ok = slicer.util.saveScene(f"{O}/../similarity_demo.mrb"); print("saved scene", ok)
