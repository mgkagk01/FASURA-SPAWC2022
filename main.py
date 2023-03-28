# -*- coding: utf-8 -*-
# Python 3.8
import numpy as np
from Fasura import FASURA
import time
import sys



np.random.seed(0)
# =========================================== Initialization =========================================== #
# Number of active users
K = 100
# number of pilots
nPilots = 896
# Spreading sequence length
L = 9
# Length of the code
nc = 512
# Length of the message, Length of the First and Second part of the message
B, Bf = 100, 16
Bs = B - Bf
# Number of spreading sequence/ pilot sequence
J = 2 ** Bf
# Length of List (Decoder)
nL = 64
NOPICE = 1

nChanlUses = int((nc / np.log2(4))*L + nPilots)
if nChanlUses > 3200:
    print("The length of the channel input is larger than 3200")
    sys.exit()


# Number of Antennas
M = 50
# EbN0dB
EbN0dB = -10

# Number of iterations
nIter = 3

# To save the results
resultsFA = np.zeros(nIter)  # For False Alarm
resultsDE = np.zeros(nIter) # For Detection

# =========================================== Simulation =========================================== #
start = time.time()
print("# ============= Simulation ============= #")
print('Number of users: ' + str(K))
print('Number of antennas: ' + str(M))
print("Number of channel uses: " +str(nChanlUses))
print("Number of iterations: "+ str(nIter))
print("Eb/N0 = " + str(EbN0dB) + " dB")
print(" Length of spreading sequence " + str(L))
print("Length of the code " +str(nc))
print("Length of message bits = " +str(B))
print("Bf = " + str(Bf))
print("NOPICE = " + str(NOPICE))
# --- Variance of Noise
sigma2 = nChanlUses / (B * (10 ** (EbN0dB / 10.0)))
print('Sigma^2: ' + str(sigma2))
count = 0

# --- Run the simulation nIter times
for Iter in range(nIter):

    if np.mod(Iter,10) == 0:
        print()
        print('======= Iteration number of the Monte Carlo Simulation: ' + str(Iter) + " =======")
        print()

    # --- Generate K msgs
    msgs = np.random.randint(0, 2, (K, B))

    # --- Generate the channel coefficients
    H = (1 / np.sqrt(2)) * (np.random.normal(0, 1, (K, M)) + 1j * np.random.normal(0, 1, (K, M)))

    # --- Create a FASURA object
    #scheme = FASURA(K, M, B, Bf, L, nc, nL, nPilots, sigma2)
    scheme = FASURA(K, M, B, Bf, L, nc, nL, nPilots, sigma2, NOPICE)
    # --- Encode the data
    XH = scheme.transmitter(msgs, H)

    # --- Generate the noise
    N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (nChanlUses, M)) + 1j * np.random.normal(0, np.sqrt(sigma2), (nChanlUses, M)))

    # --- Received Signal
    Y = XH + N

    # --- Decode
    DE, FA, Khat = scheme.receiver(Y)

    resultsFA[Iter] = FA/Khat
    resultsDE[Iter] = DE/K

print('Missed Detection')
print(sum(1-resultsDE) / float(nIter))
print('False Alarm')
print(sum(resultsFA) / float(nIter))
print()

print('Total time')
print(time.time() - start)
print()
