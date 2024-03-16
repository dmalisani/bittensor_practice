# miner.py

The miner receives a number and returns a hash of that number.

# validator.py

It generates a set of hashes to validate the miners' responses and uses a seed to generate the set. The bulk is saved to be used later. If it is called with a --seed parameter different from the last generated one, the set is regenerated using the new seed.

# Reward:
It's binary; if the response is valid, the miner will receive a complete amount, otherwise, it will receive 0.