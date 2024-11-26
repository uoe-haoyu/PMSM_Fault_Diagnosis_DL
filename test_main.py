from dataset import get_pathdata
import test
import model_repo
import os



if __name__ == '__main__':


    testdata_path = r'./Data/test_data.npy'
    os.makedirs('./Inference_test/', exist_ok=True)
    save_path = r'./Inference_test/'  # save predicted results


    # Proposed
    pth = r'./trained_model_logs/Experiment2_Proposed_two_branch_Adam_0.005/epoch76.pth'
    netname = model_repo.Proposed_two_branch

    # Convformer
    # pth = r'./trained_model_logs/Experiment1_Convformer_NSE_Adam_0.005/epoch68.pth'
    # netname = model_repo.Convformer_NSE

    # MCNN_CA
    # pth = r'./trained_model_logs/Experiment5_MCNN_CA_Adam_0.001/epoch85.pth'
    # netname = model_repo.MCNN_CA

    # RNN
    # pth = r'./trained_model_logs/Experiment6_FDGRU_Adam_0.005/epoch98.pth'
    # netname = model_repo.FDGRU


    config = {
        'netname': netname.Net,
        'dataset': {'test': get_pathdata(testdata_path),},
        'pth_repo': pth,
        'test_path': save_path,
    }

    tester = test.Test(config)
    accuracys = []

    for i in range(1):
        print(f"Running test iteration {i + 1}...")
        accuracy = tester.start()
        accuracy = accuracy*100
        accuracys.append(accuracy)
        print(f"Test iteration {i + 1} completed.")

