from pathlib import Path
import json, shutil, zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_ensemble_five_slice_logit_stack"
OUT = ROOT / "results/full_five_slice_logit_stack_submission"

MODEL = '''from pathlib import Path
import torch
import torch.nn as nn
from .densenet3d import DenseNet121_3D
from .resnet3d import ResNet18CBAM3D
class CnnPair(nn.Module):
    def __init__(self):
        super().__init__(); self.resnet=ResNet18CBAM3D(); self.densenet=DenseNet121_3D()
    def forward(self,x):
        return torch.softmax(self.resnet(x),1)[:,1], torch.softmax(self.densenet(x),1)[:,1]
def load_model(path: Path, device: torch.device):
    p=torch.load(path,map_location=device,weights_only=True); m=CnnPair(); m.resnet.load_state_dict(p["resnet_state_dict"]); m.densenet.load_state_dict(p["densenet_state_dict"]); return m.to(device).eval()
'''

PREPROCESS = '''import numpy as np
import torch
from scipy import ndimage
from .adaptive_localizer import adaptive_localization
TARGET_SHAPE=(16,64,64)
def prepare_registered(nifti_path,fixed_template):
    import ants
    moving=ants.image_read(str(nifti_path)); result=ants.registration(fixed=fixed_template,moving=moving,type_of_transform="Rigid",verbose=False)
    return np.asarray(result["warpedmovout"].numpy(),dtype=np.float64)
def prepare_tensor_from_registered(registered,channel_means,channel_stds):
    best,_=adaptive_localization(registered); start,end=int(best["start"]),int(best["end"]); valid=np.isfinite(registered)&(registered>0)
    if not valid.any(): raise RuntimeError("EMPTY_POSITIVE_REGISTERED_VOLUME")
    pos=registered[valid]; mean,std=float(pos.mean()),float(pos.std())
    if not np.isfinite(std) or std<1e-8: raise RuntimeError("INVALID_VOLUME_STANDARDIZATION")
    normalized=np.zeros(registered.shape,dtype=np.float32); normalized[valid]=((registered[valid]-mean)/std).astype(np.float32)
    slices=np.stack([normalized[:,:,z].T for z in range(start,end+1)],axis=0)
    standardized=(slices-np.asarray(channel_means,dtype=np.float32)[:,None,None])/np.asarray(channel_stds,dtype=np.float32)[:,None,None]
    resized=ndimage.zoom(standardized,tuple(TARGET_SHAPE[i]/standardized.shape[i] for i in range(3)),order=1)
    return torch.from_numpy(resized.astype(np.float32,copy=False)).unsqueeze(0).unsqueeze(0)
'''

MAIN = '''from pathlib import Path
import ants,json,numpy as np,pandas as pd,torch
from src.model import load_model
from src.preprocessing import prepare_registered,prepare_tensor_from_registered
from src.svm_features import extract_features
from src.svm_model import RbfSvm
DATA_ROOT=Path("/code_execution/data"); NIFTI_DIR=DATA_ROOT/"niftis"; SUBMISSION_FORMAT_PATH=DATA_ROOT/"submission_format.csv"; WRITE_SUBMISSION_PATH=Path("submission.csv"); ROOT=Path(__file__).resolve().parent
def main():
    sub=pd.read_csv(SUBMISSION_FORMAT_PATH); std=json.loads((ROOT/"assets/standardization.json").read_text()); md=json.loads((ROOT/"assets/model_metadata.json").read_text()); fixed=ants.image_read(str(ROOT/"assets/template.nii")); brain=ants.image_read(str(ROOT/"assets/brain_mask.nii.gz")).numpy()>0; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cnn=load_model(ROOT/"assets/model/full_ensemble_weights.pt",dev); svm=RbfSvm(ROOT/"assets/model/morphometric_intensity_rbf_svm.npz"); st=md["logit_stack"]; coef=np.asarray(st["coef"]); base=np.asarray([st["base_weights"][k] for k in ("resnet","densenet","svm")]); out=[]
    with torch.inference_mode():
        for uid in sub["uid"]:
            reg=prepare_registered(NIFTI_DIR/f"{uid}.nii.gz",fixed); ten=prepare_tensor_from_registered(reg,std["channel_means"],std["channel_stds"]); rp,dp=cnn(ten.to(dev)); f,_=extract_features(reg,brain); probs=np.asarray([float(rp.cpu()),float(dp.cpu()),svm.predict_probability(f)]); z=np.log(np.clip(probs,1e-6,1-1e-6)/(1-np.clip(probs,1e-6,1-1e-6))); score=float(np.sum(coef*base*z)+st["intercept"]); out.append(float(np.clip(1/(1+np.exp(-np.clip(score,-40,40))),.005,.995)))
    sub["is_pathologic"]=out; sub.to_csv(WRITE_SUBMISSION_PATH,index=False)
if __name__=="__main__": main()
'''

def main():
    if not PACKAGE.exists(): shutil.copytree(ROOT/"submission_ensemble_sigma075_8slice_svm",PACKAGE)
    (PACKAGE/"src/model.py").write_text(MODEL); (PACKAGE/"src/preprocessing.py").write_text(PREPROCESS); (PACKAGE/"main.py").write_text(MAIN)
    (PACKAGE/"assets/standardization.json").write_text((ROOT/"original_submission_files/standardization.json").read_text())
    md=json.loads((OUT/"training_metadata.json").read_text()); (PACKAGE/"assets/model_metadata.json").write_text(json.dumps(md,indent=2)+"\n")
    model_dir=PACKAGE/"assets/model"; (model_dir/"ensemble_weights.pt").unlink(missing_ok=True); (model_dir/"morphometric_intensity_rbf_svm.joblib").unlink(missing_ok=True)
    shutil.copy2(OUT/"full_ensemble_weights.pt",model_dir/"full_ensemble_weights.pt"); shutil.copy2(OUT/"morphometric_intensity_rbf_svm.npz",model_dir/"morphometric_intensity_rbf_svm.npz")
    archive=ROOT/"submission_ensemble_five_slice_logit_stack.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                z.write(path, path.relative_to(PACKAGE))
    print(archive)
if __name__=="__main__": main()
