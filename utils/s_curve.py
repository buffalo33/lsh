import matplotlib.pyplot as plt
import numpy as np

def generate_s_curve_data(n_division,computed_simi_matrix,complete_simi_matrix,max_simi):
    mat=np.nan_to_num(computed_simi_matrix)
    mat_jacc_ref=np.nan_to_num(complete_simi_matrix)
    count=[]
    x0=[]
    max_simi=0.7 # there are no similarities above 0.7 in the dataset
    for k in range(n_division):
        b1=k*max_simi/n_division
        b2=(k+1)*max_simi/n_division
        # count the number of elements in the computed simi matrix that have simi in the range
        simi_in_interval = np.where(((mat>b1) & (mat<b2)))[0]
        count_simi_interval=len(simi_in_interval)
        # count the number of ones in the complete simi matrix
        simi_in_interval_true = np.where(((mat_jacc_ref>b1) & (mat_jacc_ref<b2)))[0]
        count_simi_interval_true=len(simi_in_interval_true)
        if count_simi_interval_true!=0:
            if count_simi_interval>count_simi_interval_true:
                print("We gotta problm", count_simi_interval,count_simi_interval_true)
            count.append(count_simi_interval/count_simi_interval_true)
        else:
            count.append(0)
        x0.append(b1)
    return x0,count

def generate_s_curve_plot(n_division,computed_simi_matrix,complete_simi_matrix,max_simi,title="S curve"):  
    fig=plt.figure(figsize=(8,5))
    x0,count=generate_s_curve_data(n_division,computed_simi_matrix,complete_simi_matrix,max_simi)
    plt.plot(x0,count)
    plt.xlabel("Similarity")
    plt.ylabel("Probability of sharing a bucket")
    return fig