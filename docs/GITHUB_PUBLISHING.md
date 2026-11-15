# Publishing to GitHub

```bash
cd churn-guard-ai
git init
git add .
git commit -m "ChurnGuard AI: end-to-end churn prediction system"
git branch -M main
# create an empty repo named churn-guard-ai on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/churn-guard-ai.git
git push -u origin main
```

Before pushing:
1. Put your name/institution in the report: `python scripts/build_all.py --author "..." --institution "..."`.
2. If you trained on the real dataset, regenerate everything (`python -m src.train` then `build_all.py`)
   so no SYNTHETIC banners/figures remain. Do not commit the raw dataset (already git-ignored).
3. Add screenshots you captured yourself to `assets/screenshots/` and link them in the README.
4. Choose and add a LICENSE file (e.g. MIT) if you want others to reuse the code.
5. Optional: add a GitHub Actions workflow running `python -m unittest discover -s tests -t .`.
