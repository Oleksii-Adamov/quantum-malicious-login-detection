from use_case import BethEntry
import pandas as pd
import joblib

model_dir = './model'

def beth_entry_to_df(beth_entry):

    entry_data = {
        'timestamp': beth_entry.timestamp,
        'processId': beth_entry.process_id,
        'threadId': beth_entry.thread_id,
        'parentProcessId': beth_entry.parent_process_id,
        'userId': beth_entry.user_id,
        'mountNamespace': beth_entry.mount_namespace,
        'processName': beth_entry.process_name,
        'hostName': beth_entry.host_name,
        'eventId': beth_entry.event_id,
        'argsNum': len(beth_entry.args),
        'eventName': beth_entry.event_name,
        'stackAddresses': beth_entry.stack_addresses,
        'returnValue': beth_entry.return_value,
        'args': [], # not handling
    }
    
    # Convert the dictionary into a pandas DataFrame
    beth_entry_df = pd.DataFrame([entry_data])
    
    return beth_entry_df

def manual_min_max_scaling(data, min_value, max_value, feature_range):

    min_range, max_range = feature_range
    scaled_data = (data - min_value) / (max_value - min_value) * (max_range - min_range) + min_range
    return scaled_data

def preprocess_test_data(df):

    df = df.copy()
    eventIdMap = joblib.load(model_dir + '/eventIdMap.pkl')
    threadIdMap = joblib.load(model_dir + '/threadIdMap.pkl')
    # process columns as suggested by the paper http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf
    df['isOSprocessId'] = (df['processId'] < 3).astype('int')
    df['isParentOSprocessId'] = (df['parentProcessId'] < 3).astype('int')
    df['isDefaultMount'] = (df['mountNamespace'] == 4026531840).astype('int')
    df['isOSUser'] = (df['userId'] < 1000).astype('int')
    df.loc[df['returnValue'] < 0, "returnValue"] = -1
    df.loc[df['returnValue'] > 0, "returnValue"] = 1
    # might add back: threadId processName stackAddresses args
    df = df.drop(columns=['timestamp', 'processId', 'processName', 'parentProcessId', 'userId', 'mountNamespace', 'hostName', 'eventName', 'stackAddresses', 'args'])

    # frequency encode eventId
    eventIds = df['eventId'].unique()
    for eventId in eventIds:
        if not eventId in eventIdMap.keys():
            eventIdMap[eventId] = 0
    
    df['eventId'] = df['eventId'].map(eventIdMap)

    # frequency encode processName
    thread_unq = df['threadId'].unique()
    for thread in thread_unq:
        if not thread in threadIdMap.keys():
            threadIdMap[thread] = 0
    
    df['threadId'] = df['threadId'].map(threadIdMap)


    df = df.astype(float)
    # scale argsNum
    df['argsNum'] = manual_min_max_scaling(df['argsNum'], min_value=0, max_value=5, feature_range=(-1, 1))
    
    return df

def predict(X_test) -> bool:

    nystroem = joblib.load(model_dir + '/nystroem_kernel_8000_threadId_v100.pkl')
    sgd_ocsvm = joblib.load(model_dir + '/sgd_ocsvm_model_8000_threadId_v100_ac.pkl')

    X_test_transformed = nystroem.transform(X_test)

    y_pred = sgd_ocsvm.predict(X_test_transformed)

    return y_pred[0] == -1

async def solution(input: BethEntry) -> bool:

    beth_entry_df = beth_entry_to_df(input)

    beth_entry_df_pr = preprocess_test_data(beth_entry_df)

    return predict(beth_entry_df_pr)
