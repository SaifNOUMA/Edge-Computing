#%% 
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt

def f(data):
    return sorted(data,key=lambda x: x[0])

data1 = [[5e-05, 0.7014, 0.003438413143157959, 3428], [0.0001, 0.6961, 0.003339189028739929, 3242], [0.0005, 0.6821, 0.0028870212316513062, 2823], [0.001, 0.6756, 0.0029183838605880735, 2625], [0.05, 0.6332, 0.0025983633279800414, 1431], [0.1, 0.6272, 0.0023144875764846804, 1213], [0.15, 0.6212, 0.0021965712785720825, 1066], [0.2, 0.6176, 0.0021072022676467896, 943], [0.3, 0.6107, 0.0020934361934661864, 769], [0.4, 0.6043, 0.0019908159255981445, 617], [0.5, 0.6004, 0.001955173420906067, 473], [0.6, 0.595, 0.0017551474094390869, 338], [0.7, 0.5871, 0.0016538300752639771, 83], [0.8, 0.5858, 0.001630326533317566, 45], [0.9, 0.585, 0.0015850917100906372, 29], [1.0, 0.5842, 0.0015484279155731201, 14], [0.0, 0.7926, 0.006141886186599731, 10000]]
data2 = [[0.01, 0.6527, 0.03908066494464874, 1925], [0.005, 0.6589, 0.04294048099517822, 2126], [0.001, 0.6756, 0.039973819398880006, 2625], [0.0005, 0.6821, 0.03899989314079284, 2823], [5e-05, 0.7014, 0.05200949740409851, 3428], [1e-05, 0.7119, 0.04812604975700378, 3820], [5e-06, 0.7178, 0.04242972657680511, 4002], [1e-06, 0.7272, 0.04679633452892303, 4338], [1e-07, 0.7401, 0.04202933497428894, 4862]]
# cpus=50% && Memory=1Gb
data3 = f([[1e-15, 0.7819, 0.04198928039073944, 7476], [1e-14, 0.7788, 0.04220843665599823, 7231], [1e-13, 0.7747, 0.04227828941345215, 6957], [1e-12, 0.7725, 0.04486587285995483, 6682], [1e-11, 0.7676, 0.050034217882156375, 6342], [1e-10, 0.7622, 0.06230877313613892, 6026], [1e-09, 0.756, 0.05178455107212067, 5677], [1e-08, 0.7487, 0.05070750770568848, 5282], [1e-07, 0.7401, 0.22440346622467042, 4862], [1e-06, 0.7272, 0.04213218071460724, 4338], [1e-05, 0.7119, 0.030450124979019166, 3820], [0.0001, 0.6961, 0.034390604972839356, 3242], [0.001, 0.6756, 0.04762880239486694, 2625], [0.01, 0.6527, 0.04658944892883301, 1925], [0.1, 0.6272, 0.08977871730327606, 1213], [1, 0.5842, 0.04809344549179077, 14]])
# cpus=100% && Memory=2Gb
data4 = f([[1e-15, 0.7819, 0.007136275410652161, 7476], [1e-14, 0.7788, 0.0067528513908386234, 7231], [1e-13, 0.7747, 0.0065582676887512206, 6957], [1e-12, 0.7725, 0.006482267951965332, 6682], [1e-11, 0.7676, 0.0066311502695083615, 6342], [1e-10, 0.7622, 0.006324788045883179, 6026], [1e-09, 0.756, 0.005671465396881103, 5677], [1e-08, 0.7487, 0.005559672164916992, 5282], [1e-07, 0.7401, 0.005236580085754394, 4862], [1e-06, 0.7272, 0.004890253710746765, 4338], [1e-05, 0.7119, 0.004579768538475037, 3820], [0.0001, 0.6961, 0.004471665906906128, 3242], [0.001, 0.6756, 0.004357675290107727, 2625], [0.01, 0.6527, 0.003496318769454956, 1925], [0.1, 0.6272, 0.0031168116569519045, 1213], [1, 0.5842, 0.0025216240882873536, 14]])
# cpus=70% && Memory=2Gb
data5 = f([[1e-15, 0.7819, 0.029580854201316835, 7476], [1e-14, 0.7788, 0.02739660565853119, 7231], [1e-13, 0.7747, 0.026926648831367493, 6957], [1e-12, 0.7725, 0.02076594922542572, 6682], [1e-11, 0.7676, 0.027952819323539733, 6342], [1e-10, 0.7622, 0.027743753862380982, 6026], [1e-09, 0.756, 0.02623490254878998, 5677], [1e-08, 0.7487, 0.02549175021648407, 5282], [1e-07, 0.7401, 0.022656329822540282, 4862], [1e-06, 0.7272, 0.028444725513458252, 4338], [1e-05, 0.7119, 0.025394371247291565, 3820], [0.0001, 0.6961, 0.026985734581947328, 3242], [0.001, 0.6756, 0.024163757634162904, 2625], [0.01, 0.6527, 0.0197934889793396, 1925], [0.1, 0.6272, 0.022572185850143433, 1213], [1, 0.5842, 0.032159075093269346, 14]])
# cpus=100% && Memory=32Gb
data6 = f([[1e-15, 0.7819, 0.004643661046028137, 7476], [1e-14, 0.7788, 0.004801906490325927, 7231], [1e-13, 0.7747, 0.004427378392219543, 6957], [1e-12, 0.7725, 0.004326468515396118, 6682], [1e-11, 0.7676, 0.004199665760993958, 6342], [1e-10, 0.7622, 0.004078960490226746, 6026], [1e-09, 0.756, 0.003959596300125122, 5677], [1e-08, 0.7487, 0.0038006200551986696, 5282], [1e-07, 0.7401, 0.0036396652936935424, 4862], [1e-06, 0.7272, 0.0034544193267822265, 4338], [1e-05, 0.7119, 0.003254237222671509, 3820], [0.0001, 0.6961, 0.003103839683532715, 3242], [0.001, 0.6756, 0.0030997021198272707, 2625], [0.01, 0.6527, 0.002599291372299194, 1925], [0.1, 0.6272, 0.002299438214302063, 1213], [1, 0.5842, 0.0018493565559387208, 14]])
# cpus=100% && Memory=3Gb
data7 = f([[1e-15, 0.7819, 0.004810870718955994, 7476], [1e-14, 0.7788, 0.004959058308601379, 7231], [1e-13, 0.7747, 0.004733904314041137, 6957], [1e-12, 0.7725, 0.004643592166900635, 6682], [1e-11, 0.7676, 0.004541906142234802, 6342], [1e-10, 0.7622, 0.0040714328289031985, 6026], [1e-09, 0.756, 0.004067852330207825, 5677], [1e-08, 0.7487, 0.003919476652145386, 5282], [1e-07, 0.7401, 0.003741239285469055, 4862], [1e-06, 0.7272, 0.0035390830993652345, 4338], [1e-05, 0.7119, 0.0035718502521514895, 3820], [0.0001, 0.6961, 0.003240955924987793, 3242], [0.001, 0.6756, 0.002870117712020874, 2625], [0.01, 0.6527, 0.0027796510219573974, 1925], [0.1, 0.6272, 0.002188073301315308, 1213], [1, 0.5842, 0.0019196993112564086, 14]])


# cpus=100% && Memory=512Mb
data8 = f([[1e-15, 0.7819, 0.00742087996006012, 7476], [1e-14, 0.7788, 0.006780428075790405, 7231], [1e-13, 0.7747, 0.006882257413864136, 6957], [1e-12, 0.7725, 0.006224478363990784, 6682], [1e-11, 0.7676, 0.005985975289344788, 6342], [1e-10, 0.7622, 0.005803210401535034, 6026], [1e-09, 0.756, 0.00556494402885437, 5677], [1e-08, 0.7487, 0.005336607623100281, 5282], [1e-07, 0.7401, 0.005104889559745788, 4862], [1e-06, 0.7272, 0.0048125128746032714, 4338], [1e-05, 0.7119, 0.004513783025741577, 3820], [0.0001, 0.6961, 0.004203071784973145, 3242], [0.001, 0.6756, 0.003828165578842163, 2625], [0.01, 0.6527, 0.0034969818830490112, 1925], [0.1, 0.6272, 0.003061647081375122, 1213], [1, 0.5842, 0.0025174073696136476, 14]])
# cpus=100% && Memory=1Gb
data9 = f([[1e-15, 0.7819, 0.00634780797958374, 7476], [1e-14, 0.7788, 0.0060905330419540405, 7231], [1e-13, 0.7747, 0.005819849801063538, 6957], [1e-12, 0.7725, 0.00547872588634491, 6682], [1e-11, 0.7676, 0.005184047675132752, 6342], [1e-10, 0.7622, 0.0050675213098526, 6026], [1e-09, 0.756, 0.005048593688011169, 5677], [1e-08, 0.7487, 0.0049701640605926516, 5282], [1e-07, 0.7401, 0.004474056434631348, 4862], [1e-06, 0.7272, 0.004662457942962646, 4338], [1e-05, 0.7119, 0.004225303959846496, 3820], [0.0001, 0.6961, 0.0036886251449584962, 3242], [0.001, 0.6756, 0.003641612482070923, 2625], [0.01, 0.6527, 0.0034490734100341798, 1925], [0.1, 0.6272, 0.0029344229221343996, 1213], [1, 0.5842, 0.0024357664823532106, 14]])
# cpus=100% && Memory=2Gb
data10 = f([[1e-15, 0.7819, 0.0056890282392501835, 7476], [1e-14, 0.7788, 0.005750845241546631, 7231], [1e-13, 0.7747, 0.006076250100135803, 6957], [1e-12, 0.7725, 0.006008658790588379, 6682], [1e-11, 0.7676, 0.005235655426979065, 6342], [1e-10, 0.7622, 0.004420851993560791, 6026], [1e-09, 0.756, 0.004357015061378479, 5677], [1e-08, 0.7487, 0.0040393070936203005, 5282], [1e-07, 0.7401, 0.0038152924060821532, 4862], [1e-06, 0.7272, 0.0034422822952270508, 4338], [1e-05, 0.7119, 0.0032642326831817626, 3820], [0.0001, 0.6961, 0.0031400166273117067, 3242], [0.001, 0.6756, 0.003113886833190918, 2625], [0.01, 0.6527, 0.0026272264003753664, 1925], [0.1, 0.6272, 0.002291105532646179, 1213], [1, 0.5842, 0.0019880696535110472, 14]])
# cpus=100% && Memory=3Gb
data11 = f([[1e-15, 0.7819, 0.004810870718955994, 7476], [1e-14, 0.7788, 0.004959058308601379, 7231], [1e-13, 0.7747, 0.004733904314041137, 6957], [1e-12, 0.7725, 0.004643592166900635, 6682], [1e-11, 0.7676, 0.004541906142234802, 6342], [1e-10, 0.7622, 0.0040714328289031985, 6026], [1e-09, 0.756, 0.004067852330207825, 5677], [1e-08, 0.7487, 0.003919476652145386, 5282], [1e-07, 0.7401, 0.003741239285469055, 4862], [1e-06, 0.7272, 0.0035390830993652345, 4338], [1e-05, 0.7119, 0.0035718502521514895, 3820], [0.0001, 0.6961, 0.003240955924987793, 3242], [0.001, 0.6756, 0.002870117712020874, 2625], [0.01, 0.6527, 0.0027796510219573974, 1925], [0.1, 0.6272, 0.002188073301315308, 1213], [1, 0.5842, 0.0019196993112564086, 14]])

threshold = [i[0] for i in data8]
# accuracy  = [100*i[1] for i in data]
# exec_time = [1000*i[2] for i in data]
# exit_ech  = [i[3]/100 for i in data]

accuracy  = [list(map(lambda x: float(format(x,'.2f')),[100*i[1] for i in data])) for data in [data4,data5,data6,]]
exec_time = [list(map(lambda x: float(format(x,'.2f')),[1000*i[2] for i in data])) for data in [data8,data9,data10,data11]]
exit_ech  = [list(map(lambda x: float(format(x,'.2f')),[i[3]/100 for i in data])) for data in [data4,data5,data6]]

#%%  Performance result
fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(threshold, accuracy, color=color, marker='.')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Runtime (ms)', color=color)  # we already handled the x-label with ax1
ax2.plot(threshold, exec_time, color=color, marker='*')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("../results/plots/performance_results_3.png")
plt.close(fig)

#%% Samples Exited from first branch
fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Accuracy (%)', color=color)
ax1.plot(threshold, accuracy, color=color, marker='.')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Samples Exited from 1st Branch (%)', color=color)  # we already handled the x-label with ax1
ax2.plot(threshold, exit_ech, color=color, marker='*')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("../results/plots/samples_exited_from_1_branch_2.png")
plt.close(fig)


#%%

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#%% Accuracy
labels = ["1e-%d" % (i) for i in range(15,0,-1)] + ["1"]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax1 = plt.subplots(figsize=(18,12))
rects1 = ax1.bar(x - width/2, accuracy[0], width, 
                label='Accuracy')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('(%)')
ax1.set_xlabel('Entropy')
ax1.set_title('Effect on the Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

autolabel(rects1,ax1)

fig.tight_layout()
plt.savefig("../results/plots/accuracy1.png")
plt.show()

# %% Samples Exited from 1st Branch
labels = ["1e-%d" % (i) for i in range(15,0,-1)] + ["1"]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax1 = plt.subplots(figsize=(15,9))
rects1 = ax1.bar(x - width/2, exit_ech[0], width, 
                label=['Samples Exited'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('(%)')
ax1.set_xlabel('Entropy')
ax1.set_title('Effect on the Execution Location')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

autolabel(rects1,ax1)

fig.tight_layout()
plt.savefig("../results/plots/samples_exited1.png")
plt.show()

# %% Execution Time
labels = ["1e-%d" % (i) for i in range(15,0,-1)] + ["1"]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots(figsize=(18,12))

# rects1 = ax1.bar(x - width/2, exec_time[0], width, 
#                 label=['CPU (70%) & Memory=2Gb'])
rects2 = ax1.bar(x - width/2, exec_time[-2], width, 
                label=['CPU (100%) & Memory=3Gb'])
rects3 = ax1.bar(x + width/2, exec_time[-1], width, 
                label=['CPU (100%) & Memory=512Mb'])
# rects4 = ax1.bar(x + width/2, exec_time[3], width, 
#                 label=['CPU (100%) & Memory=32Gb'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('runtime (ms)')
ax1.set_xlabel('Entropy')
ax1.set_title('Effect on the Runtime')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

# autolabel(rects1,ax1)
# autolabel(rects2,ax1)
autolabel(rects2,ax1)
autolabel(rects3,ax1)

fig.tight_layout()
plt.savefig("../results/plots/runtime4.png")
plt.show()


#%%
fig, ax = plt.subplots(figsize=(18,12))

ind = np.arange(len(threshold))
width = 1        # the width of the bars
N = 4 # Number of classes

a = ax.bar(ind, 
        exec_time[0], 
        width = width/N, 
        bottom = 0,
        label = 'CPU (100%) & Memory=512Mb')

b = ax.bar(ind + width / N, 
        exec_time[1], 
        width = width/N, 
        bottom = 0,
        label = 'CPU (100%) & Memory=1Gb')

c = ax.bar(ind + 2 * width / N, 
        exec_time[2], 
        width = width/N, 
        bottom = 0,
        label = 'CPU (100%) & Memory=2Gb')

d = ax.bar(ind + 3 * width / N, 
        exec_time[3], 
        width = width/N, 
        bottom = 0,
        label = 'CPU (100%) & Memory=3Gb')

autolabel(a,ax)
autolabel(b,ax)
autolabel(c,ax)
autolabel(d,ax)
ax.set_title('Effect of the threshold on the Runtime')
ax.set_xticks(ind + (N-1) * width / (2*N))
ax.set_xticklabels(threshold)
ax.set_ylabel('Runtime (ms)')
ax.set_xlabel('Threshold')
ax.legend()
ax.yaxis.set_units(1)
ax.autoscale_view()

plt.savefig("../results/plots/runtime5.png")
plt.show()

# %%
