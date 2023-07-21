import tensorflow as tf
import numpy as np
import time
import os
import math
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from networks import u_net, T_net, Tu_net, error_u, derror_u, derror_Tu
from norms import b_norm
from gan import GAN
from callbacks import alternate_training
from analytic import u_analytic_fn, Tu_analytic_fn
import analytic
from save import save_evaluation, save_loss
import settings

# define the name of the directory to be created
dir = 'numerical_results/' + input("Numerical experiments folder name: ")
try:
    os.makedirs(dir)
except:
    raise Exception("Directory {} already exists. Try again!".format(dir))

start_time = time.time()
u = u_net(depth=2, width=20, activation='tanh')
T = T_net(depth=2, width=20, activation='tanh')
Tu = Tu_net(u, T)
e_u = error_u(u, u_analytic_fn)
de_u = derror_u(e_u)
e_Tu = error_u(Tu, Tu_analytic_fn)
de_Tu = derror_Tu(e_Tu)
model = GAN(u, Tu)
u_opt = tf.keras.optimizers.Adam()
T_opt = tf.keras.optimizers.Adam()
model.compile(u_opt, T_opt)

num_samples=40
x_plot = np.arange(0., 1.+1/num_samples-10**(-5), 1/num_samples)
x = tf.constant(np.expand_dims(x_plot, axis=1))

alternate_training_callback = alternate_training()
call=[alternate_training_callback]

data = tf.experimental.numpy.empty(shape=[2,3])
history = model.fit(data, epochs=settings.num_iterations, batch_size=settings.integration_sample_size, callbacks=call)

end_time = time.time()

x_analytic = analytic.x
u_plot = save_evaluation(x_analytic, u, "u_net", dir+"/u_net.csv")
_ = save_evaluation(x, u, "u_net", dir+"/u_net_plot.csv")
Tu_plot = save_evaluation(x_analytic, Tu, "Tu_net", dir+"/Tu_net.csv")
_ = save_evaluation(x, Tu, "Tu_net", dir+"/Tu_net_plot.csv")
u_analytic = save_evaluation(x_analytic, u_analytic_fn, "u_analytic", dir+"/u_analytic.csv")
u_analytic_plot = save_evaluation(x, u_analytic_fn, "u_analytic_plot", dir+"/u_analytic_plot.csv")
Tu_analytic = save_evaluation(x_analytic, Tu_analytic_fn, "Tu_analytic", dir+"/Tu_analytic.csv")
Tu_analytic_plot = save_evaluation(x, Tu_analytic_fn, "Tu_analytic", dir+"/Tu_analytic_plot.csv")
e_u_plot = save_evaluation(x_analytic, e_u, "e_u", dir+"/e_u.csv")
_ = save_evaluation(x, e_u, "e_u", dir+"/e_u_plot.csv")
de_u_plot = save_evaluation(x_analytic, de_u, "de_u", dir+"/de_u.csv")
_ = save_evaluation(x, de_u, "de_u", dir+"/de_u_plot.csv")
e_Tu_plot = save_evaluation(x_analytic, e_Tu, "e_Tu", dir+"/e_Tu.csv")
_ = save_evaluation(x, e_Tu, "e_Tu", dir+"/e_Tu_plot.csv")
de_Tu_plot = save_evaluation(x_analytic, de_Tu, "de_Tu", dir+"/de_Tu.csv")
_ = save_evaluation(x, de_Tu, "de_Tu", dir+"/de_Tu_plot.csv")


optimal_loss_u = analytic.optimal_loss_u
optimal_loss_T = analytic.optimal_loss_T
energy_e_u = b_norm(e_u)(x_analytic)
energy_u = analytic.optimal_u_norm
energy_relative_u = energy_e_u / energy_u
energy_e_Tu = b_norm(e_Tu)(x_analytic)
energy_Tu = analytic.optimal_Tu_norm
energy_relative_Tu = energy_e_Tu / energy_Tu

lossu_history = history.history["loss_u"]
lossT_history = history.history["loss_T"]
save_loss(lossu_history, lossT_history, dir+"/loss.csv")
iterations_indexes = range(len(lossu_history))
lossu_relative = tf.sqrt(2*tf.abs(lossu_history[-1]-optimal_loss_u))/energy_u
lossT_relative = tf.sqrt(2*tf.abs(lossT_history[-1]-optimal_loss_T))/energy_Tu

with open('settings.py', 'r') as firstfile, open(dir+'/overview.txt', 'w') as secondfile:
    # read content from first file
    for line in firstfile:
        # append content to second file
        secondfile.write(line)
    secondfile.write("\noptimal_loss_u = {0:.16f}".format(optimal_loss_u))
    secondfile.write("\nlast_loss_u = {0:.16f}".format(lossu_history[-1]))
    secondfile.write("\nmean_last_100_losses_u = {0:.16f}".format(sum(lossu_history[-100:])/100))
    secondfile.write("\noptimal_loss_T = {0:.16f}".format(optimal_loss_T))
    secondfile.write("\nlast_loss_T = {0:.16f}".format(lossT_history[-1]))
    secondfile.write("\nmean_last_100_losses_T = {0:.16f}".format(sum(lossT_history[-100:])/100))
    secondfile.write("\nrelative_error_u_energy = {0:.16f}".format(energy_relative_u))
    secondfile.write("\nrelative_error_Tu_energy = {0:.16f}".format(energy_relative_Tu))
    secondfile.write("\nrelative_lossu = {0:.16f}".format(lossu_relative))
    secondfile.write("\nrelative_lossT = {0:.16f}".format(lossT_relative))
    secondfile.write("\nexecution_time = {0:.16f}".format(end_time-start_time))

    firstfile.close()
    secondfile.close()


pdf = PdfPages(dir+'/results.pdf')

fig = plt.figure(constrained_layout=True, figsize=(11.69, 8.27))
fig.suptitle(r"MIN-MIN example: Predictions and errors --- $V:=U;\; b(u,v):=\int_0^1 u' v';\; l(v):=\int_0^1 f v;\; \Vert \cdot \Vert_V := \sqrt{b(\cdot,\cdot)} =: \Vert \cdot\Vert_U$")
gs = gridspec.GridSpec(2, 2, figure=fig)

ax_sol_test = fig.add_subplot(gs[0, 0])
ax_sol_trial = fig.add_subplot(gs[0, 1])
ax_sol_test.set_title(r'Test function')
ax_sol_trial.set_title(r'Trial function')
ax_sol_test.set_xlabel("$x$")
ax_sol_trial.set_xlabel("$x$")
ax_sol_trial.plot(x_plot, u_analytic_plot, "o-", linewidth=1.5, label=r"$u^*$", color="black")
ax_sol_trial.plot(x_analytic, u_plot, '-', linewidth=1.5, label=r"$u_h$", color="blue")
ax_sol_test.plot(x_plot, Tu_analytic_plot, "o-", linewidth=1.5, label=r"$Tu^*$", color="black")
ax_sol_test.plot(x_analytic, Tu_plot, '-', linewidth=1.5, label=r"$T_h u_h$", color="purple")
ax_sol_trial.legend()
ax_sol_test.legend()


ax_error = fig.add_subplot(gs[1, 0])
ax_error.set_title("Error")
ax_error.set_xlabel(r"$x$")
ax_error.plot(x_analytic, e_u_plot, "-", linewidth=1.5, color="red", label=r"$u^* - u_h$")
ax_error.plot(x_analytic, e_Tu_plot, "-", linewidth=1.5, color="magenta", label=r"$T u^* - T_h u_h$")
ax_error.plot(x_analytic, energy_e_u*np.ones_like(x_analytic), linestyle="dashed", linewidth=1.5, color="red", label=r"$\Vert u^* - u_h\Vert_U$")
ax_error.plot(x_analytic, energy_e_Tu*np.ones_like(x_analytic), linestyle="dashed", linewidth=1.5, color="magenta", label=r"$\Vert T u^* - T_h u_h\Vert_V$")
ax_error.legend()

fig.show()
pdf.savefig(fig)

optimal_loss_u = analytic.optimal_loss_u
optimal_loss_T = analytic.optimal_loss_T


fig = plt.figure(constrained_layout=True, figsize=(11.69, 8.27))
fig.suptitle("MIN-MIN example: Losses evolution --- {} data, {} samples/iteration".format("constant" if settings.is_data_constant else "random", settings.integration_sample_size, settings.T_max_iterations, settings.u_max_iterations))
gs = gridspec.GridSpec(3, 2, figure=fig)

lossu_history = history.history["loss_u"]
lossT_history = history.history["loss_T"]
iterations_indexes = range(len(lossu_history))

ax_lossT_zoom = fig.add_subplot(gs[0, 0])
ax_lossT_zoom.set_title(r'$\ell_T = \frac{1}{2}\Vert T_{h}u_{h}\Vert^2_V - b(u_{h}, T_{h}u_{h})$ evolution')
#ax_lossT_zoom.set_xlabel("iteration")
i=0
while i<10*(settings.T_max_iterations+settings.u_max_iterations):
    ax_lossT_zoom.plot(iterations_indexes[i:i+settings.T_max_iterations], lossT_history[i:i+settings.T_max_iterations], linewidth=0.8, marker="None", color="purple", label=r"minimizing $\ell_T$")
    ax_lossT_zoom.plot(iterations_indexes[i+settings.T_max_iterations-1:i+settings.T_max_iterations+settings.u_max_iterations+1], lossT_history[i+settings.T_max_iterations-1:i+settings.T_max_iterations+settings.u_max_iterations+1], linewidth=0.8,  color="blue", label=r"minimizing $\ell_u$")
    i=i+settings.T_max_iterations+settings.u_max_iterations
ax_lossT_zoom.set_yscale('symlog', linthresh=10**(-5))
ax_lossT_zoom.legend([r"minimizing $\ell_T$", r"minimizing $\ell_u$"])

ax_lossu_zoom = fig.add_subplot(gs[0, 1])
ax_lossu_zoom.set_title(r'$\ell_u = \frac{1}{2}\Vert T_{h}u_{h}\Vert^2_V - l(T_{h}u_{h})$ evolution')
#ax_lossu_zoom.set_xlabel("iteration")
i=0
while i<10*(settings.T_max_iterations+settings.u_max_iterations):
    ax_lossu_zoom.plot(iterations_indexes[i:i+settings.T_max_iterations], lossu_history[i:i+settings.T_max_iterations], linewidth=0.8, marker="None", color="purple", label=r"minimizing $\ell_T$")
    ax_lossu_zoom.plot(iterations_indexes[i+settings.T_max_iterations-1:i+settings.T_max_iterations+settings.u_max_iterations+1], lossu_history[i+settings.T_max_iterations-1:i+settings.T_max_iterations+settings.u_max_iterations+1], linewidth=0.8, color="blue", label=r"minimizing $\ell_u$")
    i=i+settings.T_max_iterations+settings.u_max_iterations
ax_lossu_zoom.set_yscale('symlog', linthresh=10**(-5))
ax_lossu_zoom.legend([r"minimizing $\ell_T$", r"minimizing $\ell_u$"])

ax_lossT = fig.add_subplot(gs[1, 0])
#ax_lossT.set_title(r'$\ell_T = \frac{1}{2}||T_{net}u_{net}||^2_V - l(T_{net}u_{net})$ evolution')
#ax_lossT.set_xlabel("iteration")
ax_lossT.set_yscale('symlog', linthresh=10**(-5))
ax_lossT.plot(iterations_indexes, lossT_history, linewidth=1.2, marker="None", color="gray", label=r"$\ell_T$")
ax_lossT.plot(iterations_indexes, optimal_loss_u*np.ones_like(iterations_indexes), linestyle="dashed", linewidth=0.8, marker="None", color="red", label=r"optimal $\ell_T$")
ax_lossT.legend()

ax_lossu = fig.add_subplot(gs[1, 1])
#ax_lossu.set_title(r'$\ell_u = \frac{1}{2}||T_{net}u_{net}||^2_V - b(u_{net}, T_{net}u_{net})$ evolution')
#ax_lossu.set_xlabel("iteration")
ax_lossu.set_yscale('symlog', linthresh=10**(-5))
ax_lossu.plot(iterations_indexes, lossu_history, linewidth=1.2, marker="None", color="gray", label=r"$\ell_u$")
ax_lossu.plot(iterations_indexes, optimal_loss_u*np.ones_like(iterations_indexes), linestyle="dashed", linewidth=0.8, marker="None", color="red", label=r"optimal $\ell_u$")
ax_lossu.legend()

ax_lossT_zoom_final = fig.add_subplot(gs[2, 0])
#ax_lossT_zoom_final.set_title(r'$\ell_T = \frac{1}{2}||T_{net}u_{net}||^2_V - l(T_{net}u_{net})$ evolution')
ax_lossT_zoom_final.set_xlabel("iteration")
ax_lossT_zoom_final.plot(iterations_indexes[-int(0.5*settings.num_iterations):], lossT_history[-int(0.5*settings.num_iterations):], linewidth=0.8, marker="None", color="gray", label=r"$\ell_T$")
ax_lossT_zoom_final.plot(iterations_indexes[-int(0.5*settings.num_iterations):], optimal_loss_T*np.ones_like(iterations_indexes[-int(0.5*settings.num_iterations):]), linestyle="dashed", linewidth=0.8, marker="None", color="red", label=r"optimal $\ell_T$")
ax_lossT_zoom_final.legend()

ax_lossu_zoom_final = fig.add_subplot(gs[2, 1])
#ax_lossu_zoom_final.set_title(r'$\ell_T = \frac{1}{2}||T_{net}u_{net}||^2_V - l(T_{net}u_{net})$ evolution')
ax_lossu_zoom_final.set_xlabel("iteration")
ax_lossu_zoom_final.plot(iterations_indexes[-int(0.5*settings.num_iterations):], lossu_history[-int(0.5*settings.num_iterations):], linewidth=0.8, marker="None", color="gray", label=r"$\ell_u$")
ax_lossu_zoom_final.plot(iterations_indexes[-int(0.5*settings.num_iterations):], optimal_loss_u*np.ones_like(iterations_indexes[-int(0.5*settings.num_iterations):]), linestyle="dashed", linewidth=0.8, marker="None", color="red", label=r"optimal $\ell_u$")
ax_lossu_zoom_final.legend()

fig.show()
pdf.savefig(fig)

pdf.close()