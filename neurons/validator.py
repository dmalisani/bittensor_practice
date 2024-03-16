# The MIT License (MIT)
# Copyright © 2023 Yuma Rao


import os
import time
import torch
import traceback
import argparse
import pickle
import hashlib
import random
import bittensor as bt

import template
from template.validator import forward

from template.base.validator import BaseValidatorNeuron

TEST_SET_FILENAME = "validationset.sav"


def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type = int, help='Seed to be used in random generation.')
    parser.add_argument('--validation_lot', default=10, help='Length of validation set.')
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )

    
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config =  bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'validator',
        )
    )

    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)

    return config


def generate_key_value_pairs(seed, N):
    random.seed(seed)
    key_value_pairs = {}
    for i in range(N):
        key = random.randint(0, 10000)
        value = hashlib.sha256(str(key).encode()).hexdigest()
        key_value_pairs[key] = value
    return key_value_pairs


def validation_set(*, seed: int|None, length:int|None) -> dict[int, str]:
    filename = TEST_SET_FILENAME
    if not os.path.exists(filename):
        bt.logging.info("Reading validation test")
        key_value_pairs = generate_key_value_pairs(seed, length)
        with open(filename, 'wb') as file:
            pickle.dump((seed, key_value_pairs), file)
        return key_value_pairs

    with open(filename, 'rb') as file:
        saved_seed, key_value_pairs = pickle.load(file)
        bt.logging.debug(f"saved seed: {saved_seed}")

    if seed and saved_seed != seed:
        bt.logging.info("Seed has changed, regenerating set")
        key_value_pairs = generate_key_value_pairs(seed, length)
        with open(filename, 'wb') as file:
            pickle.dump((seed, key_value_pairs), file)       

    return key_value_pairs



def main(config):
    validation_hash = validation_set(seed=config.seed, length=config.validation_lot)        
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
    bt.logging.info(config)

    bt.logging.info("Setting up bittensor objects.")

    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    dendrite = bt.dendrite( wallet = wallet )
    bt.logging.info(f"Dendrite: {dendrite}")

    metagraph = subtensor.metagraph( config.netuid )
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again.")
        exit()
    else:
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")


    bt.logging.info("Building validation weights.")
    alpha = 0.9
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")

    bt.logging.info("Starting validator loop.")

    for number, hash in validation_hash.items():
        try:
            bt.logging.info(f"sending {number} to hash")
            responses = dendrite.query(
                metagraph.axons,
                template.protocol.ToHash(nounce_input=number),
                deserialize = False, 
            )

            bt.logging.info(f"Received responses: {responses}")

            for i, resp_i in enumerate(responses):
                score = 0
                bt.logging.debug(f"Response: {repr(resp_i)}")
                if  resp_i.generated_hash == hash:
                    score = 1
                bt.logging.debug(f"Score: {score}")
                scores[i] = alpha * score + (1 - alpha) * 0

                if i % 2 == 0:
                    weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                    bt.logging.info(f"Setting weights: {weights}")

                    result = subtensor.set_weights(
                        netuid = config.netuid,
                        wallet = wallet,
                        uids = metagraph.uids,
                        weights = weights,
                        wait_for_inclusion = True
                    )
                    if result: bt.logging.success('Successfully set weights.')
                    else: bt.logging.error('Failed to set weights.') 

            metagraph = subtensor.metagraph(config.netuid)
            time.sleep(bt.__blocktime__)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()


if __name__ == "__main__":
    config = get_config()
    main( config )
    