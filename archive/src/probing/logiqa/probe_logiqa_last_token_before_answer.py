"""
probe_logiqa_pos98.py
──────────────────────
Ablation: POSITION 98 — "last token before answer"
Position 98 = second to last of 100 evenly-spaced positions.
For long sequences this is deep in reasoning, for short sequences
may still be near the answer region.
Sampled activations only, LogReg + MLP.
"""

import os, glob, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project"
SAMPLED_DIR  = os.path.join(BASE_DIR, "activations/logiqa_v3/sampled")
RESULTS_DIR  = os.path.join(BASE_DIR, "results/probing/logiqa_v3/last_token_before_answer")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES=None; HIDDEN_DIM=2560; NUM_LAYERS=37; NUM_CLASSES=4
BATCH_SIZE=64; EPOCHS=10; TEST_SIZE=0.2; RANDOM_STATE=42
LABEL_MAP={"A":0,"B":1,"C":2,"D":3}; COLORS=["#E69F00","#56B4E9"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_df(n_samples=None):
    dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(BASE_DIR,"activations/logiqa_v3/master_log_worker*.csv"))]
    df = pd.concat(dfs,ignore_index=True).drop_duplicates('example_id')
    df = df[df['resumed']==False].reset_index(drop=True)
    df = df[df['ground_truth'].isin(['A','B','C','D'])]
    df['label'] = df['ground_truth'].map(LABEL_MAP)
    def pt_valid(row):
        sp = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
        return os.path.exists(sp) and os.path.getsize(sp) > 1024
    df = df[df.apply(pt_valid,axis=1)].reset_index(drop=True)
    print(f"Valid examples: {len(df)}")
    if n_samples: df = df.head(n_samples)
    return df

class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(HIDDEN_DIM),nn.Linear(HIDDEN_DIM,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,NUM_CLASSES))
    def forward(self,x): return self.net(x)

def compute_metrics(T,P):
    return {"acc":accuracy_score(T,P),"balanced_acc":balanced_accuracy_score(T,P),"f1_macro":f1_score(T,P,average='macro',zero_division=0)}

def train_mlp(model,X_tr,y_tr,cw):
    ds=TensorDataset(torch.tensor(X_tr,dtype=torch.float32),torch.tensor(y_tr,dtype=torch.long))
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32).to(device))
    opt=optim.Adam(model.parameters(),lr=1e-3)
    model.train()
    for _ in range(EPOCHS):
        for bx,by in loader:
            opt.zero_grad(); loss=crit(model(bx.to(device)),by.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

@torch.no_grad()
def eval_mlp(model,X_te,y_te):
    model.eval()
    P=model(torch.tensor(X_te,dtype=torch.float32).to(device)).cpu().argmax(dim=-1).numpy()
    return compute_metrics(y_te,P)

df = load_df(N_SAMPLES)
train_df,test_df = train_test_split(df,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=df['label'])
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
counts = train_df['label'].value_counts().sort_index()
cw = [len(train_df)/(NUM_CLASSES*counts.get(i,1)) for i in range(NUM_CLASSES)]

print("Loading caches...")
train_cache={row['sampled_tensor_file']:torch.load(os.path.join(SAMPLED_DIR,row['sampled_tensor_file']),map_location='cpu') for _,row in train_df.iterrows() if os.path.exists(os.path.join(SAMPLED_DIR,row['sampled_tensor_file']))}
test_cache={row['sampled_tensor_file']:torch.load(os.path.join(SAMPLED_DIR,row['sampled_tensor_file']),map_location='cpu') for _,row in test_df.iterrows() if os.path.exists(os.path.join(SAMPLED_DIR,row['sampled_tensor_file']))}
print(f"Train: {len(train_cache)} | Test: {len(test_cache)}")

all_results=[]
for layer in range(NUM_LAYERS):
    print(f"\n── Layer {layer}/{NUM_LAYERS-1} ──",flush=True)
    X_tr,y_tr=[],[]
    for _,row in train_df.iterrows():
        act=train_cache.get(row['sampled_tensor_file'])
        if act is None: continue
        X_tr.append(act[layer,98,:].float().numpy()); y_tr.append(int(row['label']))
    X_te,y_te=[],[]
    for _,row in test_df.iterrows():
        act=test_cache.get(row['sampled_tensor_file'])
        if act is None: continue
        X_te.append(act[layer,98,:].float().numpy()); y_te.append(int(row['label']))
    X_tr,y_tr=np.array(X_tr),np.array(y_tr)
    X_te,y_te=np.array(X_te),np.array(y_te)

    rr={"layer":layer}
    clf=LogisticRegression(max_iter=200,class_weight='balanced',random_state=RANDOM_STATE,solver='saga',C=1.0)
    clf.fit(X_tr,y_tr); m=compute_metrics(y_te,clf.predict(X_te))
    rr["logreg_acc"]=m["acc"]; rr["logreg_balanced_acc"]=m["balanced_acc"]; rr["logreg_f1_macro"]=m["f1_macro"]
    print(f"  LogReg → Acc: {m['acc']:.4f} | Bal Acc: {m['balanced_acc']:.4f} | F1: {m['f1_macro']:.4f}",flush=True)

    mlp=MLPProbe().to(device); train_mlp(mlp,X_tr,y_tr,cw); m=eval_mlp(mlp,X_te,y_te)
    rr["mlp_acc"]=m["acc"]; rr["mlp_balanced_acc"]=m["balanced_acc"]; rr["mlp_f1_macro"]=m["f1_macro"]
    print(f"  MLP    → Acc: {m['acc']:.4f} | Bal Acc: {m['balanced_acc']:.4f} | F1: {m['f1_macro']:.4f}",flush=True)

    all_results.append(rr)
    pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_DIR,"results_sampled.csv"),index=False)

results_df=pd.DataFrame(all_results); layers=results_df['layer']
for metric,ylabel,fname,baseline,bl_label in [
    ("balanced_acc","Balanced Accuracy","balanced_acc",0.25,"Chance (25%)"),
    ("acc","Accuracy","accuracy",0.30,"Majority baseline"),
]:
    plt.figure(figsize=(12,5))
    plt.plot(layers,results_df[f'logreg_{metric}'],marker='o',color=COLORS[0],label='LogReg')
    plt.plot(layers,results_df[f'mlp_{metric}'],marker='s',color=COLORS[1],label='MLP')
    plt.axhline(y=baseline,color='gray',linestyle='--',alpha=0.5,label=bl_label)
    plt.title(f"Layer-wise {ylabel} — pos98 | sampled\n(Qwen3-4B on LogiQA v3)")
    plt.xlabel("Layer"); plt.ylabel(ylabel); plt.ylim(0,1.05)
    plt.xticks(range(0,NUM_LAYERS,2)); plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"{fname}_sampled.png"),dpi=300); plt.close()

print(f"\nBest layer (LogReg Bal Acc): Layer {results_df.loc[results_df['logreg_balanced_acc'].idxmax(),'layer']} ({results_df['logreg_balanced_acc'].max():.4f})")
print(f"Results → {RESULTS_DIR}/")