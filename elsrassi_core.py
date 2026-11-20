from dataclasses import dataclass
import numpy as np
import pandas as pd

G_CM_S2 = 980.665

@dataclass
class SolverConfig:
    analysis_type: str = "Equivalent linear"
    input_location: str = "Outcrop bedrock"
    effective_strain_ratio: float = 0.65
    max_iterations: int = 20
    convergence_tolerance: float = 0.03
    relaxation: float = 0.5
    constitutive_model: str = "Darendeli"
    groundwater_depth_m: float = np.inf
    rock_vs_mps: float = 1000.0
    rock_density_mg_m3: float = 2.5
    rock_damping_percent: float = 2.0
    darendeli_pi: float = 0.0
    darendeli_alpha: float = 0.919
    darendeli_ocr: float = 1.0
    darendeli_n_cycles: float = 10.0
    darendeli_frequency_hz: float = 1.0
    darendeli_k0: float = 0.5
    use_layer_specific_darendeli: bool = False
    remove_input_mean: bool = True

def acceleration_to_cm_s2(x, units):
    x=np.asarray(x,float)
    return x if units=="cm/s²" else x*100 if units=="m/s²" else x*G_CM_S2

def validate_profile(p):
    p=p.copy().reset_index(drop=True)
    req=["thickness_m","Vs_mps","density_Mg_m3"]
    if any(c not in p for c in req): raise ValueError("Profile requires thickness_m, Vs_mps, density_Mg_m3")
    defaults={"damping_percent":2.,"PI":0.,"OCR":1.,"K0":0.5,"curve_id":"A"}
    for c,v in defaults.items():
        if c not in p: p[c]=v
    for c in req+["damping_percent","PI","OCR","K0"]: p[c]=pd.to_numeric(p[c],errors="raise")
    if (p["thickness_m"]<=0).any() or (p["Vs_mps"]<=0).any() or (p["density_Mg_m3"]<=0).any(): raise ValueError("Thickness, Vs and density must be positive")
    p["curve_id"]=p["curve_id"].astype(str)
    return p

def validate_custom_curves(c):
    c=c.copy(); req=["curve_id","strain_percent","G_over_Gmax","damping_percent"]
    if any(k not in c for k in req): raise ValueError("Custom curves require curve_id,strain_percent,G_over_Gmax,damping_percent")
    for k in req[1:]: c[k]=pd.to_numeric(c[k],errors="raise")
    c["curve_id"]=c["curve_id"].astype(str)
    return c.sort_values(["curve_id","strain_percent"]).reset_index(drop=True)

def stresses(profile,cfg):
    h=profile.thickness_m.to_numpy(float); rho=profile.density_Mg_m3.to_numpy(float)
    z=np.r_[0,np.cumsum(h)]; zm=z[:-1]+h/2
    sv=np.cumsum(rho*9.81*h)-rho*9.81*h/2
    u=np.zeros_like(zm) if np.isinf(cfg.groundwater_depth_m) else 9.81*np.maximum(zm-cfg.groundwater_depth_m,0)
    sve=sv-u
    k0=profile.K0.to_numpy(float) if cfg.use_layer_specific_darendeli else np.full(len(h),cfg.darendeli_k0)
    return z,zm,sve,sve*(1+2*k0)/3

def darendeli(max_strain_pct,R,sigma,PI,alpha,OCR,ncyc,freq):
    pa=101.3; eff=R*np.asarray(max_strain_pct,float)
    c1=-1.1143*alpha**2+1.8618*alpha+0.2533
    c2=0.0805*alpha**2-0.071*alpha-0.0095; c3=-0.0005*alpha**2+0.0002*alpha+0.0003
    b=0.6329-0.0057*np.log(ncyc)
    gr=(sigma/pa)**0.3483*(0.0352+0.001*PI*OCR**0.3246)
    dmin=(sigma*0.00986923267)**(-0.2889)*(0.8005+0.0129*PI*OCR**(-0.1069))*(1+0.2919*np.log(freq))
    g=1/(1+(eff/gr)**alpha); d=dmin.copy(); m=eff>1e-12
    e=eff[m]; r=gr[m]
    dm1=(100/np.pi)*(4*((e-r*np.log((e+r)/r))/(e**2/(e+r)))-2)
    dm=c1*dm1+c2*dm1**2+c3*dm1**3
    d[m]=b*g[m]**0.1*dm+dmin[m]
    return g,d

def custom_props(max_strain_pct,R,profile,curves):
    eff=R*np.asarray(max_strain_pct,float); g=np.zeros(len(profile)); d=np.zeros(len(profile))
    groups={k:v for k,v in curves.groupby("curve_id")}
    for i,cid in enumerate(profile.curve_id.astype(str)):
        if cid not in groups: raise ValueError(f"Missing custom curve {cid}")
        q=groups[cid]; x=q.strain_percent.to_numpy(float)
        g[i]=np.interp(eff[i],x,q.G_over_Gmax); d[i]=np.interp(eff[i],x,q.damping_percent)
    return g,d

def layered_response(ag,dt,h,vs,rho,damp,input_location):
    ag=np.asarray(ag,float); h=np.asarray(h,float); vs=np.asarray(vs,float); rho=np.asarray(rho,float); damp=np.asarray(damp,float)
    N=len(vs); ns=N-1; L=len(ag); AG=np.fft.fft(ag); ff=np.fft.rfftfreq(L,dt); npos=len(ff)
    az=(rho[:-1]*vs[:-1]*(1+1j*damp[:-1]))/(rho[1:]*vs[1:]*(1+1j*damp[1:]))
    E=np.zeros((L,N+1),complex); B=np.zeros_like(E); E[:,0]=B[:,0]=1; E[0,:]=B[0,:]=1
    ksh=np.zeros((npos,ns),complex)
    for i in range(1,npos):
        ksh[i]=2*np.pi*ff[i]*h/(vs[:-1]*(1+1j*damp[:-1]))
        for j in range(1,N):
            k=ksh[i,j-1]
            E[i,j]=.5*E[i,j-1]*(1+az[j-1])*np.exp(1j*k)+.5*B[i,j-1]*(1-az[j-1])*np.exp(-1j*k)
            B[i,j]=.5*E[i,j-1]*(1-az[j-1])*np.exp(1j*k)+.5*B[i,j-1]*(1+az[j-1])*np.exp(-1j*k)
        E[i,N]=E[i,N-1]; B[i,N]=E[i,N-1]
        if not (L%2==0 and i==npos-1): E[L-i]=np.conj(E[i]); B[L-i]=np.conj(B[i])
    inp=N if input_location=="Outcrop bedrock" else N-1
    F=np.zeros(L,complex); TF=np.zeros((L,N+1),complex); TS=np.zeros((L,ns),complex); F[0]=1; TF[0]=1
    for i in range(1,npos):
        den=E[i,inp]+B[i,inp]; F[i]=(E[i,0]+B[i,0])/den; TF[i]=(E[i]+B[i])/den; w=2*np.pi*ff[i]
        for j in range(ns):
            k=ksh[i,j]; TS[i,j]=(1j*k/h[j])*(E[i,j]*np.exp(1j*k/2)-B[i,j]*np.exp(-1j*k/2))/(-w*w*den)
        if L%2==0 and i==npos-1: F[i]=F[i].real; TF[i]=TF[i].real; TS[i]=TS[i].real
        else: F[L-i]=np.conj(F[i]); TF[L-i]=np.conj(TF[i]); TS[L-i]=np.conj(TS[i])
    out=np.fft.ifft(AG*F).real; at=np.fft.ifft(AG[:,None]*TF,axis=0).real
    strain=np.fft.ifft(AG[:,None]*TS,axis=0).real
    stress=(strain/100)*(rho[:-1]*vs[:-1]**2)[None,:]
    fas_in=np.abs(AG[:npos])/L; fas_out=np.abs((AG*F)[:npos])/L
    if L%2==0: fas_in[1:-1]*=2; fas_out[1:-1]*=2
    else: fas_in[1:]*=2; fas_out[1:]*=2
    return {"ff":ff,"absF":np.abs(F[:npos]),"input_fas":fas_in,"output_fas":fas_out,"surface":out,"ATIME":at,"strain":strain,"stress":stress,"max_strain":np.max(np.abs(strain),axis=0)}

def run_analysis(acc,dt,profile,cfg,custom_curves=None):
    p=validate_profile(profile); ag=np.asarray(acc,float).ravel()
    if cfg.remove_input_mean: ag=ag-ag.mean()
    h=p.thickness_m.to_numpy(float); vmax=p.Vs_mps.to_numpy(float); rhos=p.density_Mg_m3.to_numpy(float)
    z,zm,sve,sme=stresses(p,cfg); rho=np.r_[rhos,cfg.rock_density_mg_m3]; n=len(p)
    if cfg.analysis_type=="Linear":
        g=np.ones(n); dp=p.damping_percent.to_numpy(float); vs=np.r_[vmax,cfg.rock_vs_mps]
        resp=layered_response(ag,dt,h,vs,rho,np.r_[dp/100,cfg.rock_damping_percent/100],cfg.input_location); hist=[]; conv=True
    else:
        curves=validate_custom_curves(custom_curves) if cfg.constitutive_model=="Custom" else None
        PI=p.PI.to_numpy(float) if cfg.use_layer_specific_darendeli else np.full(n,cfg.darendeli_pi)
        OCR=p.OCR.to_numpy(float) if cfg.use_layer_specific_darendeli else np.full(n,cfg.darendeli_ocr)
        if curves is None: _,dp=darendeli(np.zeros(n),cfg.effective_strain_ratio,sme,PI,cfg.darendeli_alpha,OCR,cfg.darendeli_n_cycles,cfg.darendeli_frequency_hz)
        else: dp=np.array([curves[curves.curve_id==c].damping_percent.iloc[0] for c in p.curve_id.astype(str)])
        g=np.ones(n); d=dp/100; hist=[]; conv=False
        for it in range(1,cfg.max_iterations+1):
            vs=np.r_[vmax*np.sqrt(g),cfg.rock_vs_mps]
            resp=layered_response(ag,dt,h,vs,rho,np.r_[d,cfg.rock_damping_percent/100],cfg.input_location)
            if curves is None: gt,dtp=darendeli(resp["max_strain"],cfg.effective_strain_ratio,sme,PI,cfg.darendeli_alpha,OCR,cfg.darendeli_n_cycles,cfg.darendeli_frequency_hz)
            else: gt,dtp=custom_props(resp["max_strain"],cfg.effective_strain_ratio,p,curves)
            dtarget=dtp/100; eG=np.max(np.abs(gt-g)/np.maximum(np.abs(g),1e-6)); eD=np.max(np.abs(dtarget-d)/np.maximum(np.abs(d),1e-4)); hist.append((it,eG,eD))
            if max(eG,eD)<=cfg.convergence_tolerance: g,d=gt,dtarget; conv=True; break
            g=(1-cfg.relaxation)*g+cfg.relaxation*gt; d=(1-cfg.relaxation)*d+cfg.relaxation*dtarget
        dp=d*100; vs=np.r_[vmax*np.sqrt(g),cfg.rock_vs_mps]
        resp=layered_response(ag,dt,h,vs,rho,np.r_[d,cfg.rock_damping_percent/100],cfg.input_location)
    maxacc=np.max(np.abs(resp["ATIME"][:,:n+1]),axis=0); maxstress=np.max(np.abs(resp["stress"]),axis=0)
    layers=pd.DataFrame({"layer":np.arange(1,n+1),"mid_depth_m":zm,"Vs_max_mps":vmax,"Vs_final_mps":vs[:-1],"G_over_Gmax":g,"damping_percent":dp,"max_strain_percent":resp["max_strain"],"max_shear_stress_kPa":maxstress,"sigma_v_eff_kPa":sve,"sigma_m_eff_kPa":sme})
    return {"summary":{"surface_PGA_cm_s2":float(np.max(np.abs(resp["surface"]))),"input_PGA_cm_s2":float(np.max(np.abs(ag))),"converged":conv,"iterations":len(hist)},"layers":layers,"time":np.arange(len(ag))*dt,"input":ag,"surface":resp["surface"],"freq":resp["ff"],"tf":resp["absF"],"fas_in":resp["input_fas"],"fas_out":resp["output_fas"],"acc_depth":z,"max_acc":maxacc,"history":pd.DataFrame(hist,columns=["iteration","err_G","err_D"]),"interfaces":z}
