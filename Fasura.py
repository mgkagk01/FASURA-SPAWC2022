# -*- coding: utf-8 -*-

import numpy as np
from PolarCode import PolarCode
from utilities import bin2dec, dec2bin, crcEncoder, crcDecoder, QPSK, LMMSE
from tools import *
from estiFuncs import symbolsEst, channelEst, channelEstWithErrors
class FASURA():
    def __init__(self, K, M, B, Bf, L, nc, nL, nPilots, sigma2, NOPICE):
        ''' Parameters '''
        self.K = K # Number of Users
        self.M = M  # Number of antennas
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.L = L  # Length of spreading sequence
        self.J = 2 ** Bf  # Number of spreading sequence
        self.nc = nc  # length of code
        self.nL = nL  # List size
        self.nQPSKSymbols = int(nc / 2) # Number of QPSK symbols
        self.nDataSymbols = int(L * self.nQPSKSymbols)
        self.nPilots = nPilots  # number of pilot symbols
        self.nChanlUses = self.nPilots + self.nDataSymbols
        self.sigma2 = sigma2
        self.save = 0

        ''' For polar code '''
        # Polynomial for CRC coding
        if K < 10:
            self.divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
        else:
            self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = np.round(np.random.randint(low=0, high=2, size=(self.nc - self.msgLen, self.J)))

        # Create a polar Code object
        self.polar = PolarCode(self.nc, self.msgLen, self.K)

        ''' Generate matrices '''
        # Pilots
        self.P = ((1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J)))) + 1j * (
                1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J))))) / np.sqrt(2.0)

        # Spreading sequence master set
        self.A = (np.random.normal(loc=0, scale=1, size=(self.nQPSKSymbols * self.L, self.J)) + 1j * np.random.normal(
            loc=0, scale=1, size=(self.nQPSKSymbols * self.L, self.J)))

        for j in range(self.nQPSKSymbols):
            temp = np.linalg.norm(self.A[j * self.L:(j + 1) * self.L, :], axis=0)
            self.A[j * self.L:(j + 1) * self.L, :] = np.divide(self.A[j * self.L:(j + 1) * self.L, :], temp)

        self.A = (np.sqrt(self.L) * self.A)

        # Interleaver
        self.interleaver = np.zeros((self.nc, self.J), dtype=int)
        for j in range(self.J):
            self.interleaver[:, j] = np.random.choice(self.nc, self.nc, replace=False)

        ''' To store information '''
        self.msgs = np.zeros((K, Bf + self.Bs), dtype=int)  # Store the active messages
        self.msgsHat = np.zeros((K, Bf + self.Bs), dtype=int)  # Store the recovered messages

        self.idxSS = np.zeros(self.K, dtype=int)


        self.count = 0  # Count the number of recovered msgs in this round
        self.Y = np.zeros((self.nChanlUses, M))
        self.idxSSDec = np.array([], dtype=int)
        self.idxSSHat = np.array([], dtype=int)  # To store the new recovered sequences
        self.symbolsHat = np.zeros((self.K, self.nQPSKSymbols), dtype=complex)
        self.NOPICE = NOPICE
        

    def transmitter(self, msgs, H):

        '''
        Function to encode the messages of the users
        Inputs: 1. the message of the users in the binary form, dimensions of msgBin, K x B
                2. Channel
        Output: The sum of the channel output before noise, dimensions of Y, n x M
        '''

        # ===================== Initialization ===================== #
        Y = np.zeros((self.nChanlUses, self.M), dtype=complex)

        # --- Step 0: Save the messages
        self.msgs = msgs.copy()
        # --- For all active users
        for k in range(self.K):

            # --- Step 1: Break the message into two parts
            # First part, Second part
            mf = self.msgs[k, 0:self.Bf]
            ms = self.msgs[k, self.Bf::]

            # --- Step 2: Find the decimal representation of mf
            self.idxSS[k] = bin2dec(mf)

            # --- Step 3: Append CRC bits to ms
            msgCRC = crcEncoder(ms, self.divisor)

            # --- Step 4: polar encode
            codeword, _ = self.polar.encoder(msgCRC, self.frozenValues[:, self.idxSS[k]], k)

            # --- Step 5: Interleaver
            codeword = codeword[self.interleaver[:, self.idxSS[k]]]

            # --- Step 6: QPSK modulation
            symbols = QPSK(codeword)
            self.symbolsTest = symbols

            # --- For Pilots (PH)
            PH = np.kron(self.P[:, self.idxSS[k]], H[k, :]).reshape(self.nPilots, self.M)

            # --- For Symbols (QH)
            A = np.zeros((self.nDataSymbols), dtype=complex)
            for t in range(self.nQPSKSymbols):
                A[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, self.idxSS[k]] * symbols[t]

            QH = np.kron(A, H[k, :]).reshape(self.nDataSymbols, self.M)

            # --- Add the new matrix to the output signal
            Y += np.vstack((PH, QH))

        return Y

    def receiver(self, Y):

        '''
        Function to recover the messages of the users from noisy observations
        Input:  The received signal, dimensions of Y, n x M
        Output: Probability of Detection and False Alarm
        '''

        # --- Save the received signal
        self.Y = Y.copy()

        # =========================================== Receiver  =========================================== #
        while True:

            # ======================== Pilot / Spreading Sequence Detector ======================== #
            self.idxSSHat = energyDetector(self, self.Y, self.K - self.count)

            # ======================== Channel estimation (Pilots) ======================== #
            HhatNew = channelEst(self)

            # ======================== Symbol estimation and Polar Code ======================== #
            userDecRx, notUserDecRx, symbolsHatHard, msgsHat2Part = decoder(self, HhatNew, self.idxSSHat)

            # ======================== NOPICE without Polar Decoder since is done above ======================== #
            if self.NOPICE:

                # --- Estimate the channel using P and Q
                HhatNew2 = channelEstWithErrors(self, symbolsHatHard)

                # --- Symbol Estimation and polar code
                userDecRx2, notUserDecRx2, symbolsHatHard2, msgsHat2Part2 = decoder(self, HhatNew2, self.idxSSHat)

                if userDecRx2.size >= userDecRx.size:
                    userDecRx = userDecRx2
                    notUserDecRx = notUserDecRx2
                    symbolsHatHard = symbolsHatHard2
                    msgsHat2Part = msgsHat2Part2

            # --- Add the new indices
            totalNumIndices = len(self.idxSSDec) + len(self.idxSSHat[userDecRx])
            if totalNumIndices > self.K:
                diff = totalNumIndices - self.K
                userDecRx = userDecRx[0:len(userDecRx)-diff]

            self.idxSSDec = np.append(self.idxSSDec, self.idxSSHat[userDecRx])

            # ======================== Exit Condition ======================== #
            # --- No new decoded user
            if userDecRx.size == 0:
                print('=== Done ===')
                DE, FA = checkPerformance(self)
                return DE, FA, self.count

            # ======================== Channel estimation (P + Q) ======================== #
            # --- Estimate the channel of the correct users
            # Use the received signal
            self.Y = Y.copy()
            HhatNewDec = channelEstWithDecUsers(self, Y, self.idxSSDec, symbolsHatHard[userDecRx,:])

            # ================================== SIC ================================== #
            # Only one user is decoded
            if userDecRx.size == 1:
                userDecRx = userDecRx[0]
                # Only one user left
                if msgsHat2Part.shape[0] == 1:
                    if not isIncluded(self, msgsHat2Part, self.idxSSHat[userDecRx]):
                        Hsub = np.squeeze(HhatNewDec.reshape(self.M, 1))
                        subInter(self, np.ndarray.flatten(symbolsHatHard[userDecRx, :]), self.idxSSHat[userDecRx], Hsub)
                        saveUser(self, msgsHat2Part[userDecRx,:], self.idxSSHat[userDecRx])

                # More than one user left
                else:
                    if not isIncluded(self, msgsHat2Part[userDecRx, :], self.idxSSHat[userDecRx]):
                        Hsub = np.squeeze(HhatNewDec)
                        subInter(self, np.ndarray.flatten(symbolsHatHard[userDecRx, :]), self.idxSSHat[userDecRx], Hsub)
                        saveUser(self, msgsHat2Part[userDecRx, :], self.idxSSHat[userDecRx])

            # More than one user decode
            else:
                Hsub = HhatNewDec
                for g in range(userDecRx.size):
                    if not isIncluded(self, msgsHat2Part[userDecRx[g], :], self.idxSSHat[userDecRx[g]]):
                        subInter(self, symbolsHatHard[userDecRx[g]], self.idxSSHat[userDecRx[g]], Hsub[g, :])
                        saveUser(self, msgsHat2Part[userDecRx[g], :], self.idxSSHat[userDecRx[g]])

            # ======================== Find the performance ======================== #
            de, fa = checkPerformance(self)
            print('Number of Detections: ' + str(de))
            print('Number of False Alarms: ' + str(fa))
            print()

            # ======================== Exit Condition ======================== #
            if self.count == self.K or self.save == 0:
                print('=== Done ===')
                DE, FA = checkPerformance(self)
                return DE, FA, self.count
            else:
                self.save = 0


# ============================================ Functions ============================================ #
# === Energy Detector
def energyDetector(self, y, K):
    # --- Energy Per Antenna
    energy = np.linalg.norm(np.dot(self.P.conj().T, y[0:self.nPilots, :]), axis=1) ** 2

    pivot = self.nPilots
    for t in range(self.nQPSKSymbols):
        energy += np.linalg.norm(np.dot(self.A[t * self.L: (t + 1) * self.L, :].conj().T,
                                        y[pivot + t * self.L: pivot + (t + 1) * self.L, :]), axis=1) ** 2

    return (-energy).argsort()[:K]
    # return np.argpartition(energy, -K)[-K:]


# ==== Decoder
def decoder(self, H, idxSSHat):
    K = idxSSHat.size
    symbolsHatHard = np.zeros((K, self.nQPSKSymbols), dtype=complex)

    # ==================================== Symbol Estimation Decoder ============================================== #
    symbolsHat = symbolsEst(self.Y[self.nPilots::, :], H, self.A[:, idxSSHat], np.eye(K),
                            np.eye(self.L * self.M) * self.sigma2, self.nQPSKSymbols, self.L)

    # aveError(self.symbolsTest,symbolsHat)

    # ==================================== Channel Decoder ============================================== #
    userDecRx = np.array([], dtype=int)
    notUserDecRx = np.array([], dtype=int)
    msgsHat = np.zeros((K, self.Bs), dtype=int)
    c = 0
    for s in range(symbolsHat.shape[0]):
        # Form the codeword
        cwordHatSoft = np.concatenate((np.real(symbolsHat[s, :]), np.imag(symbolsHat[s, :])), 0)

        # Interleaver
        cwordHatSoftInt = np.zeros(self.nc)
        cwordHatSoftInt[self.interleaver[:, self.idxSSHat[s]]] = cwordHatSoft

        # Call polar decoder
        cwordHatHard, isDecoded, msgHat = polarDecoder(self, np.sqrt(2) * cwordHatSoftInt, self.idxSSHat[s])

        if isDecoded == 1 and sum(abs(((cwordHatSoftInt < 0) * 1 - cwordHatHard)) % 2) > self.nc / 2:
            isDecoded = 0

        symbolsHatHard[s, :] = QPSK(cwordHatHard[self.interleaver[:, self.idxSSHat[s]]])
        msgsHat[s, :] = msgHat

        if isDecoded:
            userDecRx = np.append(userDecRx, s)
        else:
            notUserDecRx = np.append(notUserDecRx, s)

    return userDecRx, notUserDecRx, symbolsHatHard, msgsHat


def polarDecoder(self, bitsHat, idxSSHat):
    # ============ Polar decoder ============ #
    msgCRCHat, PML = self.polar.listDecoder(bitsHat, self.frozenValues[:, idxSSHat], self.nL)

    # ============ Check CRC ============ #
    # --- Initialization
    thres, flag = np.Inf, -1
    isDecoded = 0

    # --- Check the CRC constraint for all message in the list
    for l in range(self.nL):
        check = crcDecoder(msgCRCHat[l, :], self.divisor)
        if check:
            # --- Check if its PML is larger than the current PML
            if PML[l] < thres:
                flag = l
                thres = PML[l]
                isDecoded = 1

    if thres == np.Inf:
        # --- Return the message with the minimum PML
        flag = np.argmin(PML)

    # --- Encode the estimated message
    codewordHat, _ = self.polar.encoder(msgCRCHat[flag, :], self.frozenValues[:, idxSSHat], -1)

    return codewordHat, isDecoded, msgCRCHat[flag, 0:self.Bs]

# === General Functions
def isIncluded(self, second, idxSS):
    # --- Convert the decimal index of the spreading sequence to the binary string of length self.Bf
    first = dec2bin(np.hstack((idxSS, idxSS)), self.Bf)
    # --- Concatenate the two parts
    msgHat = np.append(first[0, :], second)

    # --- Check if we recovered this message
    for i in range(self.count):
        # --- Binary Addition
        binSum = sum((msgHat + self.msgsHat[i, :]) % 2)

        if binSum == 0:
            return 1
    return 0


def subInter(self, symbols, idxSS, h):
    # Define a temp Matrix and fill the matrix
    YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)
    YTempSymbols = np.zeros((self.nQPSKSymbols * self.L, self.M), dtype=complex)

    # --- For Pilots
    for m in range(self.M):
        YTempPilots[:, m] = np.squeeze(self.P[:, idxSS]) * h[m]

    # --- For Symbols
    A = np.zeros((self.nQPSKSymbols * self.L), dtype=complex)
    for t in range(self.nQPSKSymbols):
        A[t * self.L: (t + 1) * self.L] = np.squeeze(self.A[t * self.L: (t + 1) * self.L, idxSS]) * symbols[t]

    for m in range(self.M):
        YTempSymbols[:, m] = A * h[m]

    # Subtract (SIC)
    self.Y -= np.vstack((YTempPilots, YTempSymbols))


def saveUser(self, msg2Part, idxSS):
    self.msgsHat[self.count, :] = np.concatenate(
        (np.squeeze(dec2bin(np.array([idxSS]), self.Bf)), np.squeeze(msg2Part)), 0)
    self.count += 1
    self.save = 1


def checkPerformance(self):
    numDE, numFA = 0, 0
    for i in range(self.count):
        flag = 0
        for k in range(self.K):
            binSum = sum((self.msgs[k, :] + self.msgsHat[i, :]) % 2)

            if binSum == 0:
                flag = 1
                break
        if flag == 1:
            numDE += 1
        else:
            numFA += 1

    return numDE, numFA



def channelEstWithDecUsers(self, Y, decUsersSS, symbolsHatHard):
    for i in range(self.count - 1, -1, -1):
        symbolsHatHard = np.vstack((self.symbolsHat[i, :], symbolsHatHard))

    K = decUsersSS.shape[0]
    # -- Create A
    A = np.zeros((self.nChanlUses, K), dtype=complex)
    for k in range(K):
        Atemp = np.zeros((self.nQPSKSymbols * self.L), dtype=complex)
        for t in range(self.nQPSKSymbols):
            if K > 1:
                Atemp[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, decUsersSS[k]] * \
                                                      symbolsHatHard[k, t]
            else:
                Atemp[t * self.L: (t + 1) * self.L] = np.squeeze(
                    self.A[t * self.L: (t + 1) * self.L, decUsersSS[k]] * symbolsHatHard[k,t])
        if K > 1:
            A[:, k] = np.hstack((self.P[:, decUsersSS[k]], Atemp))
        else:
            A[:, k] = np.hstack((self.P[:, decUsersSS[k]], Atemp))

    Hhat = LMMSE(Y, A, np.eye(K), np.eye(self.nChanlUses) * self.sigma2)

    for i in range(self.count):
        subInter(self, self.symbolsHat[i, :], decUsersSS[i], Hhat[i, :])

    return Hhat[self.count::, :]
