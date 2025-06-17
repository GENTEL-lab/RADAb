from ImmuneBuilder import ABodyBuilder2
import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import math

import numpy as np
from scipy.spatial.transform import Rotation
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        SS = np.diag([1., 1., -1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    A_aligned = R @ A + t
    rmsd = np.sqrt(np.mean(np.sum((B - A_aligned) ** 2, axis=0)))
    return A_aligned, R, t, rmsd
def reslist_rmsd_kabsch(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    # Extract CA coordinates
    coords_short = np.array([res['CA'].get_coord() for res in res_short]) #12*3
    coords_long = np.array([res['CA'].get_coord() for res in res_long])

    # Align the shorter protein to the longer one
    coords_short_aligned = rigid_transform_Kabsch_3D(coords_short.T, coords_long[:M].T )[0]
    coords_short_aligned = coords_short_aligned.T
    def d(i, j):
        coord_i = coords_short_aligned[i]
        coord_j = coords_long[j]
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)

    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([ d(i+k, j+k) for k in range(N-j) ])
    
    for j in range(N):
        SD[M-1, j] = d(M-1, j)

    for i in range(M-2, -1, -1):
        for j in range((N-(M-i))-1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i+1, j+1],
                SD[i, j+1]
            )

    min_SD = SD[0, :N-M+1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD
def entity_to_seq(entity):
    seq = ''
    mapping = []
    for res in Selection.unfold_entities(entity, 'R'):
        try:
            seq += three_to_one(res.get_resname())
            mapping.append(res.get_id())
        except KeyError:
            pass
    assert len(seq) == len(mapping)
    return seq, mapping
def reslist_rmsd(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = np.array(res_short[i]['CA'].get_coord())
        coord_j = np.array(res_long[j]['CA'].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([ d(i+k, j+k) for k in range(N-j) ])
    
    for j in range(N):
        SD[M-1, j] = d(M-1, j)

    for i in range(M-2, -1, -1):
        for j in range((N-(M-i))-1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i+1, j+1],
                SD[i, j+1]
            )

    min_SD = SD[0, :N-M+1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD

def reslist_rmsd(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = np.array(res_short[i]['CA'].get_coord())
        coord_j = np.array(res_long[j]['CA'].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([ d(i+k, j+k) for k in range(N-j) ])
    
    for j in range(N):
        SD[M-1, j] = d(M-1, j)

    for i in range(M-2, -1, -1):
        for j in range((N-(M-i))-1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i+1, j+1],
                SD[i, j+1]
            )

    min_SD = SD[0, :N-M+1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD
def scRMSD(genCDR, refCDR):
    """
    Calculate scRMSD between predicted and reference chains.
    """
    try:
        return reslist_rmsd(genCDR, refCDR)
    except Exception as e:
        print(e)
        return 100
