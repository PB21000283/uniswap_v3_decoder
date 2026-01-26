# UniSwap V3 Transaction Decoder

Decode a Uniswap V3 swap transaction on **Ethereum Mainnet (L1)** into a structured JSON output.

## Environment
- Python >= 3.9

## Set RPC_URL (Ethereum Mainnet RPC)
# The script reads the RPC endpoint from the environment variable RPC_URL.

# --- Windows PowerShell (run these lines in PowerShell, not bash) ---
```shell
$env:RPC_URL="https://YOUR_MAINNET_RPC_ENDPOINT"
```
# --- macOS / Linux (bash/zsh) ---
```shell
export RPC_URL="https://YOUR_MAINNET_RPC_ENDPOINT"
```
## Quick check
echo $RPC_URL

## Run
```shell
python uniswap_v3_decoder.py 0x7fdee03ffb227454946852b815b6b86d38e77e6190985c1816b41a8a7b790ea0
python uniswap_v3_decoder.py 0xc54a5fd6eda9d2ba36d61f7f9186501203e79796222cf6bb4a4c2191ffd59955
python uniswap_v3_decoder.py 0x333a8b6a401362b8cd2ce6dad693357d0230bbca149f757e7cd2d347f92126fe
```