Generate the residual heatmap after training:

```bash
python tools/visualize_gossip_residual.py \
  --checkpoint exp/pascal/732/exp/checkpoints/ckpt_best.pth \
  --dataset pascal \
  --output paper/figures/pascal_residual_heatmap.png
```

The paper draft references `pascal_residual_heatmap.png`. Run the command above before compiling the final PDF.
