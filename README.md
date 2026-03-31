# CS450-SPRING2026-CryptoBot
This is a group project for CS-450 at SDSU. The project is an Artificial Intelligence project with five collaborators. The main goal of the project is to train a bot using crypto data to make profitable trades automatically. 

## 🚀 Quick Start (Team Setup)
1. **Install uv:** - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **Sync Environment:**
   Run `uv sync` in the root folder. (This automatically installs Python 3.12 and all libraries).
3. **Run Dashboard:**
   Run `uv run streamlit run app.py`

## Project Structure

├CS450-SPRING2026-CryptoBot/
├── .devcontainer/           # VS Code container
├── data/                    # Data Storage (All .csv ignored by Git)
│   ├── logs/                
│   ├── processed/           
│   └── raw/                 
├── docs/                  
├── outputs/                 
├── src/                     
│   ├── bot/                 
│   ├── dashboard/           
│   └── data/                
│   └── model/               
├── .dockerignore            # Standardizes what stays out of the image
├── .env                     # Your private API Keys
├── .gitignore               # Standardizes what stays out of git repo
├── .python-version          # Pins project to Python 3.12
├── app.py                   # Main Entry Point (Streamlit)
├── Dockerfile               # Linux build instructions
├── docker-compose.yml       # One-command launch
├── pyproject.toml           # Dependency list
├── uv.lock                  # Version blueprint
├── README.md                # Team setup guide
└── app.py                   # Main Entry Point (Streamlit)
