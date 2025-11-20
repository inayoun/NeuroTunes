# NeuroTunes

NeuroTunes demonstrates an EEG-driven engagement index across the OpenMIIR dataset and publishes a static exploration site.

## Demo Build Workflow

```bash
python ei-core/compute_ei.py --data_dir openmiir/eeg --subject auto --out_dir ei-core/out
cp ei-core/out/ai_clips.json docs/data/
cp ei-core/out/ai_windows.csv docs/data/
node tools/build_spotify.js
git add docs/data && git commit -m "update demo" && git push
```

## Deployment

Enable GitHub Pages with Source set to `main` and Folder set to `/docs`. Each execution of the demo build workflow updates the public site with fresh data and recommendations.