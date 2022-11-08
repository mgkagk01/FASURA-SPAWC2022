from utilities import LMMSE
import numpy as np

# ==== Functions For Symbol Estimation
def symbolsEst(Y, H, A, Qx, Qn, nSlots, L):
    '''
        Function that implements the symbols estimation for spread-based MIMO system

        X: diagonal matrix contains the symbol for each user, H: channel matrix, N: noise

        Input: Y: received vector, H: channel matrix,
               A: spreading sequence master set (for all slots, only active) dim(A) = (L x nSlots) x totalNumber of Spreading Sequence,
               Qx: covariance of x, Qn: covariance of the noise
               nSlots: number of slots which is equal to the number of symbols
               L: Length of the spreading sequence
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    '''

    K = H.shape[0]

    symbolsHat = np.zeros((K, nSlots), dtype=complex)

    # --- For all Symbols
    for t in range(nSlots):
        symbolsHat[:, t] = symbolEstSubRoutine(Y[t * L: (t + 1) * L, :], H, A[t * L:(t + 1) * L, :], Qx, Qn)

    return symbolsHat


def symbolEstSubRoutine(Y, H, S, Qx, Qn):
    '''
        Function that implements the symbol estimation for spread-based MIMO system Y = SXH + N
        where Y: received matrix, S: spreading sequence matrix (only active columns),
        X: diagonal matrix contains the symbol for each user, H: channel matrix, N: noise

        Input: Y: received vector, H: channel matrix, S: spreading sequence matrix ,
               Qx: covariance of x, Qn: covariance of the noise
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    '''

    K, M = H.shape  # K: number of users, M: number of antennas
    L = S.shape[0]  # L: length of the spreading sequence

    # --- First Step:
    # Convert the system from Y = SXH + N, to y = Ax + n, where A contains the channel and sequence

    A = np.zeros((L * M, K), dtype=complex)

    for m in range(M):
        if K == 1:
            A[m * L:L * (m + 1), :] = S * H[:, m]
        else:
            # --- Diagonalize H
            A[m * L:L * (m + 1), :] = np.dot(S, np.diag(H[:, m]))

    # --- Second Step:

    # Flat Y
    y = np.ndarray.flatten(Y.T)

    # Estimate the symbols
    return LMMSE(y, A, Qx, Qn)

# ==== Functions For Channel Estimation
def channelEst(self):
    return LMMSE(self.Y[0:self.nPilots, :], self.P[:, self.idxSSHat], np.eye(len(self.idxSSHat)),
                 np.eye(self.nPilots) * self.sigma2)


def channelEstWithErrors(self, symbolsHatHard):
    K = symbolsHatHard.shape[0]

    # -- Create A
    A = np.zeros((self.nChanlUses, K), dtype=complex)
    for k in range(K):
        Atemp = np.zeros((self.nQPSKSymbols * self.L), dtype=complex)
        for t in range(self.nQPSKSymbols):
            Atemp[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, self.idxSSHat[k]] * \
                                                  symbolsHatHard[k, t]

        A[:, k] = np.hstack((self.P[:, self.idxSSHat[k]], Atemp))

    return LMMSE(self.Y, A, np.eye(K), np.eye(self.nChanlUses) * self.sigma2)

