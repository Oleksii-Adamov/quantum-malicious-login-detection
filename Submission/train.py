import pandas as pd
import time
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM

def preprocess_data(df):

    df = df.copy()
    # process columns as suggested by the paper http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf
    df['isOSprocessId'] = (df['processId'] < 3).astype('int')
    df['isParentOSprocessId'] = (df['parentProcessId'] < 3).astype('int')
    df['isDefaultMount'] = (df['mountNamespace'] == 4026531840).astype('int')
    df['isOSUser'] = (df['userId'] < 1000).astype('int')
    df.loc[df['returnValue'] < 0, "returnValue"] = -1
    df.loc[df['returnValue'] > 0, "returnValue"] = 1
    
    df = df.drop(columns=['timestamp', 'processId', 'parentProcessId', 'userId', 'mountNamespace', 'processName', 'hostName', 'eventName', 'stackAddresses', 'args'])

    # delete 'sus' entries
    df = df[df['sus'] == 0]
    
    # drop labels
    df = df.drop(columns=['sus', 'evil'])
    
    # frequency encode eventId
    eventIdMap = df['eventId'].value_counts(normalize=True)
    joblib.dump(eventIdMap, 'model/eventIdMap.pkl')
    df['eventId'] = df['eventId'].map(eventIdMap)

    # frequency encode threadId
    threadIdMap = df['threadId'].value_counts(normalize=True)
    joblib.dump(threadIdMap, 'model/threadIdMap.pkl')
    df['threadId'] = df['threadId'].map(threadIdMap)


    df = df.astype(float)
    # scale argsNum
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pd.DataFrame(df['argsNum']))
    df['argsNum'] = scaler.transform(pd.DataFrame(df['argsNum']))

    return df



def train():
    start = time.time()
    print("preprocessing")
    train_df = pd.read_csv('data/training.csv')
    pr_train_df = preprocess_data(train_df)
    n_samples = 8000
    X_train = shuffle(pr_train_df, random_state=0, n_samples=n_samples)

    feature_map = ZZFeatureMap(feature_dimension=pr_train_df.shape[1], reps=2, entanglement="linear")
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map, evaluate_duplicates='none')
    print("Nystroem")
    n_components = 100
    nystroem = Nystroem(kernel=quantum_kernel.evaluate, n_components=n_components, random_state=0)
    nystroem.fit(X_train)
    X_train_transformed = nystroem.transform(X_train)
    print("SGD")
    sgd_ocsvm = SGDOneClassSVM(max_iter=100, tol=1e-9, nu=0.015246464646464647, random_state=0, learning_rate='optimal', average = False)
    sgd_ocsvm.fit(X_train_transformed)

    joblib.dump(nystroem, 'model/nystroem_kernel_8000_threadId_v100.pkl')
    joblib.dump(sgd_ocsvm, 'model/sgd_ocsvm_model_8000_threadId_v100_ac.pkl')
    end = time.time()
    print("Training finished, execution time [sec]: ", end - start)

if __name__ == "__main__":
    train()