source /etc/profile.d/clash.sh
proxy_on

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="$no_proxy"

cd /kaggle/MABe/models
# kaggle models update -p ./models
kaggle models instances versions create \
  thekog/xgb-models-new/Other/xgb-models-new \
  -p . -n "N_SAMPLES=4_500_000"
# kaggle models instances create -p .
