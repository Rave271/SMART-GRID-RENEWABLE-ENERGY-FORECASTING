"""Smart Grid Renewable Energy Forecasting - FastAPI Application."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.config import FRONTEND_DIR, DEFAULT_SAMPLE_PATH, validate_auth_config
from app.db.database import init_db
from app.routes import upload, weather, train, predict, recommendations, reports


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize database and verify sample data on startup."""
    validate_auth_config()
    init_db()
    # Check if sample data exists; if not, generate it
    if not DEFAULT_SAMPLE_PATH.exists():
        from generate_sample_data import generate_smart_grid_data
        df = generate_smart_grid_data()
        DEFAULT_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DEFAULT_SAMPLE_PATH, index=False)

    yield


app = FastAPI(
    title="Smart Grid Renewable Energy Forecasting API",
    description="A complete ML platform forecasting solar/wind generation and electricity demand with smart dispatch recommendations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend and external integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(upload.router)
app.include_router(weather.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(recommendations.router)
app.include_router(reports.router)

# Mount static frontend assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve Dashboard Single Page Application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Smart Grid API is running. Visit /docs for Swagger documentation."}


@app.get("/signin", include_in_schema=False)
async def serve_signin():
    """Serve the presentation-only sign-in screen."""
    return HTMLResponse(
        """
        <!DOCTYPE html><html lang="en" data-theme="light"><head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sign in · Smart Grid Forecasting</title><link rel="stylesheet" href="/static/css/style.css">
        </head><body class="signin-page"><main class="signin-shell">
        <section class="signin-art"><a class="signin-brand" href="/signin"><span class="brand-icon">⚡</span><span>Smart Grid <strong>Forecasting</strong></span></a>
        <div class="signin-art-copy"><div class="eyebrow"><span class="pulse-dot"></span> Renewable intelligence platform</div>
        <h1>Make every<br><span>megawatt count.</span></h1><p>Forecast cleaner energy, understand demand, and make the next grid decision with confidence.</p></div>
        <div class="signin-art-chart" aria-hidden="true"><div class="mini-chart-line"></div><div class="mini-chart-bars"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><span class="chart-label">LIVE FORECAST SIGNAL</span></div>
        <div class="signin-art-footer"><span>Solar</span><span>Wind</span><span>Demand</span><span>Dispatch</span></div></section>
        <section class="signin-panel"><div class="signin-panel-inner"><div class="mobile-brand"><span class="brand-icon">⚡</span> Smart Grid Forecasting</div>
        <div class="signin-kicker">Welcome back</div><h2>Sign in to your workspace</h2><p class="signin-subtitle">Access the forecasting dashboard and live grid intelligence.</p>
        <form id="signin-form" class="signin-form"><label for="signin-email">Email address</label><div class="input-wrap"><span>✉</span><input id="signin-email" type="email" placeholder="you@university.edu" required></div>
        <div class="label-row"><label for="signin-password">Password</label><a href="#" id="forgot-password">Forgot password?</a></div>
        <div class="input-wrap"><span>▣</span><input id="signin-password" type="password" placeholder="Enter your password" required><button type="button" class="password-toggle" id="toggle-password">Show</button></div>
        <label class="remember-row"><input id="remember-me" type="checkbox" checked> <span>Keep me signed in on this device</span></label>
        <button type="submit" class="btn btn-primary signin-submit">Sign in <span>→</span></button><p id="signin-message" class="signin-message"></p></form>
        <div class="signin-divider"><span>Presentation demo</span></div><button id="demo-login" class="demo-login">Use demo workspace <span>↗</span></button>
        <p class="signin-note">Demo access is stored only in this browser session. API protection is configured separately through the server API key.</p></div></section></main>
        <script>
        const form=document.getElementById("signin-form"),email=document.getElementById("signin-email"),password=document.getElementById("signin-password"),message=document.getElementById("signin-message");
        function enterWorkspace(address){const session=JSON.stringify({email:address,signedInAt:new Date().toISOString()});const storage=document.getElementById("remember-me").checked?localStorage:sessionStorage;storage.setItem("sg_demo_session",session);window.location.href="/";}
        form.addEventListener("submit",e=>{e.preventDefault();if(email.value&&password.value)enterWorkspace(email.value.trim());});
        document.getElementById("demo-login").addEventListener("click",()=>{email.value="presenter@smartgrid.demo";password.value="demo-workspace";enterWorkspace(email.value);});
        document.getElementById("toggle-password").addEventListener("click",e=>{const visible=password.type==="text";password.type=visible?"password":"text";e.currentTarget.textContent=visible?"Show":"Hide";});
        document.getElementById("forgot-password").addEventListener("click",e=>{e.preventDefault();message.textContent="For this presentation demo, use the demo workspace button.";});
        </script></body></html>
        """
    )
