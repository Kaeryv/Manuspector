python ms_predictor.py -savgol 0 0 -action train \
  -max_samples 6 -lr {{lr}} -rho {{rho}} \
  -validation historical+ -noise {{noise}} -seed {{seed}} \
  -complexity  {{complexity}} -latent {{latent}} -cwt 32 {{offset}} 1 4 \
  -dropout {{dropout}} -epochs 2 > out.txt
