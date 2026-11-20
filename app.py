import io, json, zipfile
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from elsrassi_core import SolverConfig, acceleration_to_cm_s2, run_analysis, validate_profile, validate_custom_curves
st.set_page_config(page_title="ELSRASSI Web",page_icon="🌎",layout="wide")

def profile0():
    return pd.DataFrame({"thickness_m":np.ones(30),"Vs_mps":np.r_[np.full(10,160.),np.full(10,350.),np.full(10,460.)],"density_Mg_m3":np.full(30,1.8),"damping_percent":np.full(30,2.),"PI":np.zeros(30),"OCR":np.ones(30),"K0":np.full(30,.5),"curve_id":["A"]*30})
def curves0():
    return pd.DataFrame({"curve_id":["A"]*11,"strain_percent":[.0001,.0003,.001,.003,.01,.03,.1,.3,1,3,10],"G_over_Gmax":[1,1,1,.981,.941,.847,.656,.438,.238,.144,.11],"damping_percent":[.24,.42,.8,1.4,2.8,5.1,9.8,15.5,21,25,28]})
def read_motion(f):
    s=f.getvalue().decode("utf-8",errors="ignore"); d=pd.read_csv(io.StringIO(s),sep=r"[\s,;]+",engine="python",header=None,comment="#").apply(pd.to_numeric,errors="coerce").dropna(how="all").dropna(axis=1,how="all")
    if d.shape[1] not in (1,2): raise ValueError("Motion must have 1 acceleration column or 2 columns: time, acceleration")
    return d

def step(v,z): return np.repeat(v,2),np.column_stack([z[:-1],z[1:]]).reshape(-1)
def fig_xy(x,y,xlab,ylab="Depth (m)",rev=True):
    f=go.Figure(go.Scatter(x=x,y=y,mode="lines")); f.update_xaxes(title=xlab); f.update_yaxes(title=ylab,autorange="reversed" if rev else True); return f

st.title("ELSRASSI Web"); st.caption("1D linear and equivalent-linear site response for vertically propagating SH waves")
with st.expander("1 · Input acceleration record",True):
    a,b,c=st.columns(3); f=a.file_uploader("Record (.txt/.csv/.dat)",type=["txt","csv","dat"]); units=b.selectbox("Units",["cm/s²","m/s²","g"]); scale=b.number_input("Scale factor",value=1.0); loc=c.selectbox("Input location",["Outcrop bedrock","Within bedrock"]); dt0=c.number_input("dt for 1-column records (s)",value=.005,min_value=1e-6,format="%.6g")
    acc=None; dt=dt0
    if f:
        try:
            d=read_motion(f); raw=d.iloc[:,-1].to_numpy(float)
            if d.shape[1]==2:
                t=d.iloc[:,0].to_numpy(float); dt=float(np.median(np.diff(t)))
                if np.max(np.abs(np.diff(t)-dt))>max(1e-10,1e-4*dt): st.warning("Nonuniform time column; manual dt used."); dt=dt0
            acc=acceleration_to_cm_s2(raw*scale,units); st.success(f"{len(acc):,} samples · dt={dt:.6g} s")
        except Exception as e: st.error(str(e))
with st.expander("2 · Soil profile and bedrock",True):
    pf=st.file_uploader("Profile CSV (optional)",type="csv"); seed=pd.read_csv(pf) if pf else profile0(); p=st.data_editor(seed,num_rows="dynamic",use_container_width=True)
    c1,c2,c3=st.columns(3); rv=c1.number_input("Bedrock Vs (m/s)",value=1000.); rr=c2.number_input("Bedrock density (Mg/m³)",value=2.5); rd=c3.number_input("Bedrock damping (%)",value=2.)
with st.expander("3 · Analysis settings",True):
    typ=st.radio("Analysis",["Linear","Equivalent linear"],horizontal=True,index=1); gw=st.checkbox("Include groundwater"); gwd=st.number_input("Groundwater depth (m)",value=5.,disabled=not gw)
    R=.65; it=20; tol=.03; relax=.5; model="Darendeli"; curves=None; PI=0.; OCR=1.; K0=.5; alpha=.919; ncyc=10.; freq=1.; layerpars=False
    if typ=="Equivalent linear":
        q=st.columns(4); R=q[0].number_input("Effective strain ratio R",value=.65,min_value=.01,max_value=1.); it=q[1].number_input("Max iterations",value=20,min_value=1,step=1); tol=q[2].number_input("Tolerance (%)",value=3.,min_value=.01)/100; relax=q[3].number_input("Under-relaxation",value=.5,min_value=.01,max_value=1.)
        model=st.radio("Soil curves",["Darendeli","Custom"],horizontal=True)
        if model=="Darendeli":
            q=st.columns(3); PI=q[0].number_input("PI",value=0.); OCR=q[0].number_input("OCR",value=1.); K0=q[1].number_input("K₀",value=.5); alpha=q[1].number_input("Darendeli α",value=.919); ncyc=q[2].number_input("Equivalent cycles",value=10.); freq=q[2].number_input("Loading frequency (Hz)",value=1.); layerpars=st.checkbox("Use layer-specific PI/OCR/K₀ from profile")
        else:
            cf=st.file_uploader("Custom curves CSV",type="csv"); curves=st.data_editor(pd.read_csv(cf) if cf else curves0(),num_rows="dynamic",use_container_width=True)
run=st.button("Run analysis",type="primary",use_container_width=True)
if run:
    if acc is None: st.error("Upload an acceleration record first."); st.stop()
    try:
        p=validate_profile(p); curves=validate_custom_curves(curves) if model=="Custom" else curves
        cfg=SolverConfig(analysis_type=typ,input_location=loc,effective_strain_ratio=R,max_iterations=int(it),convergence_tolerance=tol,relaxation=relax,constitutive_model=model,groundwater_depth_m=gwd if gw else np.inf,rock_vs_mps=rv,rock_density_mg_m3=rr,rock_damping_percent=rd,darendeli_pi=PI,darendeli_ocr=OCR,darendeli_k0=K0,darendeli_alpha=alpha,darendeli_n_cycles=ncyc,darendeli_frequency_hz=freq,use_layer_specific_darendeli=layerpars)
        with st.spinner("Running analysis..."): r=run_analysis(acc,dt,p,cfg,curves)
        st.session_state["r"],st.session_state["cfg"],st.session_state["p"],st.session_state["curves"]=r,cfg,p,curves
    except Exception as e: st.exception(e)
if "r" in st.session_state:
    r=st.session_state.r; s=r["summary"]; st.header("Results"); a,b,c=st.columns(3); a.metric("Surface PGA",f'{s["surface_PGA_cm_s2"]:.3f} cm/s²'); b.metric("Input PGA",f'{s["input_PGA_cm_s2"]:.3f} cm/s²'); c.metric("Convergence",f'{s["iterations"]} iter',"Converged" if s["converged"] else "Not converged")
    tabs=st.tabs(["Time history","Transfer / FAS","Profiles","Convergence","Download"])
    with tabs[0]:
        f=go.Figure(); f.add_scatter(x=r["time"],y=r["input"],name="Input"); f.add_scatter(x=r["time"],y=r["surface"],name="Surface"); f.update_layout(xaxis_title="Time (s)",yaxis_title="Acceleration (cm/s²)"); st.plotly_chart(f,use_container_width=True)
    with tabs[1]:
        c1,c2=st.columns(2); c1.plotly_chart(fig_xy(r["freq"],r["tf"],"Frequency (Hz)","Transfer amplitude",False),use_container_width=True); f=go.Figure(); f.add_scatter(x=r["freq"],y=r["fas_in"],name="Input"); f.add_scatter(x=r["freq"],y=r["fas_out"],name="Surface"); f.update_xaxes(type="log",title="Frequency (Hz)"); f.update_yaxes(type="log",title="Fourier amplitude"); c2.plotly_chart(f,use_container_width=True)
    with tabs[2]:
        L=r["layers"]; c1,c2=st.columns(2); x,y=step(L.Vs_final_mps.to_numpy(),r["interfaces"]); c1.plotly_chart(fig_xy(x,y,"Vs (m/s)"),use_container_width=True); c2.plotly_chart(fig_xy(r["max_acc"],r["acc_depth"],"Maximum acceleration (cm/s²)"),use_container_width=True)
        c3,c4=st.columns(2); c3.plotly_chart(fig_xy(L.max_shear_stress_kPa,L.mid_depth_m,"Maximum shear stress (kPa)"),use_container_width=True); c4.plotly_chart(fig_xy(L.max_strain_percent,L.mid_depth_m,"Maximum shear strain (%)"),use_container_width=True)
        if st.session_state.cfg.analysis_type=="Equivalent linear":
            c5,c6=st.columns(2); c5.plotly_chart(fig_xy(L.damping_percent,L.mid_depth_m,"Damping ratio (%)"),use_container_width=True); c6.plotly_chart(fig_xy(L.G_over_Gmax,L.mid_depth_m,"G/Gmax"),use_container_width=True)
    with tabs[3]: st.dataframe(r["history"],use_container_width=True)
    with tabs[4]:
        st.dataframe(r["layers"],use_container_width=True); bio=io.BytesIO()
        with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("profile_results.csv",r["layers"].to_csv(index=False)); z.writestr("time_history.csv",pd.DataFrame({"time_s":r["time"],"input_acc_cm_s2":r["input"],"surface_acc_cm_s2":r["surface"]}).to_csv(index=False)); z.writestr("summary.json",json.dumps(s,indent=2))
        st.download_button("Download results ZIP",bio.getvalue(),"elsrassi_results.zip","application/zip",use_container_width=True)
st.caption("Engineering/research tool. Validate project-specific design analyses independently.")
