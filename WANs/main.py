import tensorflow as tf
import numpy as np
import os
import time
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from networks import u_net, v_net, error_u, derror_u
from norms import b_norm
from gan import GAN
import settings
from callbacks import alternate_training
from analytic import u_analytic_fn
import analytic
from save import save_evaluation, save_loss

# define the name of the directory to be created
dir = 'numerical_results/' + input("Numerical experiments folder name: ")
try:
    os.makedirs(dir)
except:
    raise Exception("Directory {} already exists. Try again!".format(dir))

start_time = time.time()
u = u_net(depth=2, width=20, activation='tanh')
v = v_net(depth=2, width=20, activation='tanh')
e_u = error_u(u, u_analytic_fn)
de_u = derror_u(e_u)
model = GAN(u, v, u_trainable=False, v_trainable=True)
u_opt = tf.keras.optimizers.Adam()
v_opt = tf.keras.optimizers.Adam()
model.compile(u_opt, v_opt)

num_samples=40
x_plot = np.arange(0., 1.+1/num_samples-10**(-5), 1/num_samples)
x = tf.constant(np.expand_dims(x_plot, axis=1))

alternate_trainning_callback = alternate_training()
call=[alternate_trainning_callback]

data = tf.experimental.numpy.empty(shape=[2,3]) # This is a dummy input that is required in .fit. We feed the neural networks online with random data at each train step
history = model.fit(data, epochs=settings.num_iterations, batch_size=settings.integration_sample_size, callbacks=call)

end_time = time.time()

x_analytic = analytic.x
u_plot = save_evaluation(x_analytic, u, "u_net", dir+"/u_net.csv")
_ = save_evaluation(x, u, "u_net", dir+"/u_net_plot.csv")
v_plot = save_evaluation(x_analytic, v, "v_net", dir+"/v_net.csv")
_ = save_evaluation(x, v, "v_net", dir+"/v_net_plot.csv")
u_analytic = save_evaluation(x_analytic, u_analytic_fn, "u_analytic", dir+"/u_analytic.csv")
u_analytic_plot = save_evaluation(x, u_analytic_fn, "u_analytic_plot", dir+"/u_analytic_plot.csv")
e_u_plot = save_evaluation(x_analytic, e_u, "e_u", dir+"/e_u.csv")
_ = save_evaluation(x, e_u, "e_u", dir+"/e_u_plot.csv")
de_u_plot = save_evaluation(x_analytic, de_u, "de_u", dir+"/de_u.csv")
_ = save_evaluation(x, de_u, "de_u", dir+"/de_u_plot.csv")

optimal_loss_uv = analytic.optimal_loss_uv
energy_e_u = b_norm(e_u)(x_analytic)
energy_u = analytic.optimal_u_norm
energy_relative_u = energy_e_u / energy_u

lossuv_history = history.history["loss_uv"]
save_loss(lossuv_history, dir+"/loss.csv")
iterations_indexes = range(len(lossuv_history))
lossuv_relative = tf.sqrt(2*tf.abs(lossuv_history[-1]-optimal_loss_uv))/energy_u

with open('settings.py', 'r') as firstfile, open(dir+'/overview.txt', 'w') as secondfile:
    # read content from first file
    for line in firstfile:
        # append content to second file
        secondfile.write(line)
    secondfile.write("\noptimal_loss = {0:.16f}".format(optimal_loss_uv))
    secondfile.write("\nlast_loss = {0:.16f}".format(lossuv_history[-1]))
    secondfile.write("\nmean_last_100_losses = {0:.16f}".format(sum(lossuv_history[-100:])/100))
    secondfile.write("\nrelative_error_u_energy = {0:.16f}".format(energy_relative_u))
    secondfile.write("\nrelative_lossuv = {0:.16f}".format(lossuv_relative))
    secondfile.write("\nexecution_time = {0:.16f}".format(end_time-start_time))

    firstfile.close()
    secondfile.close()
    
# From now on, we create auxiliary matplotlib plots. It needs actualization.
pdf = PdfPages(dir+'/results.pdf')

fig = plt.figure(constrained_layout=True, figsize=(11.69, 8.27))
fig.suptitle(r"MIN-MAX example: Predictions and errors --- $V:=U;\; b(u,v):=\int_0^1 u' v';\; l(v):=\int_0^1 f v;\; \Vert \cdot \Vert_V := \sqrt{b(\cdot,\cdot)} =: \Vert \cdot\Vert_U$")
gs = gridspec.GridSpec(2, 2, figure=fig)

ax_sol_test = fig.add_subplot(gs[0, 0])
ax_sol_trial = fig.add_subplot(gs[0, 1])
ax_sol_test.set_title(r'Test function')
ax_sol_trial.set_title(r'Trial function')
ax_sol_test.set_xlabel("$x$")
ax_sol_trial.set_xlabel("$x$")
ax_sol_trial.plot(x, u_analytic_plot, "o-", linewidth=1.5, label=r"$u^*$", color="black")
ax_sol_trial.plot(x_analytic, u_plot, '-', linewidth=1.5, label=r"$u_h$", color="blue")
ax_sol_test.plot(x_analytic, v_plot, '-', linewidth=1.5, label=r"$v_h$", color="purple")
ax_sol_trial.legend()
ax_sol_test.legend()


ax_error = fig.add_subplot(gs[1, 0])
ax_error.set_title("Error")
ax_error.set_xlabel(r"$x$")
ax_error.plot(x_analytic, e_u_plot, "-", linewidth=1.5, color="red", label=r"$u^* - u_h$")
#ax_error.plot(x_analytic, e_Tu(x_analytic), "-", linewidth=1.5, color="magenta", label=r"$T u^* - T_h u_h$")
ax_error.plot(x_analytic, energy_e_u*np.ones_like(x_analytic), linestyle="dashed", linewidth=1.5, color="red", label=r"$\Vert u^* - u_h\Vert_U$")
#ax_error.plot(x_analytic, energy_e_Tu*np.ones_like(x_analytic), linestyle="dashed", linewidth=1.5, color="magenta", label=r"$\Vert T u^* - T_h u_h\Vert_V$")
ax_error.legend()

fig.show()
pdf.savefig(fig)

optimal_loss_uv = analytic.optimal_loss_uv

fig = plt.figure(constrained_layout=True, figsize=(11.69, 8.27))
fig.suptitle("MIN-MAX example: Losses evolution --- {} data, {} samples/iteration".format("constant" if settings.is_data_constant else "random", settings.integration_sample_size))
gs = gridspec.GridSpec(3, 1, figure=fig)

ax_lossuv_zoom = fig.add_subplot(gs[0, 0])
ax_lossuv_zoom.set_title(r'$\ell = |b(u_h, v_h)-l(v_h)|/\Vert v_h\Vert_V$ evolution')
#ax_lossT_zoom.set_xlabel("iteration")
i=0
while i<10*(settings.v_max_iterations+settings.u_max_iterations):
    ax_lossuv_zoom.plot(iterations_indexes[i:i+settings.v_max_iterations], lossuv_history[i:i+settings.v_max_iterations], linewidth=1.2, marker="None", color="purple", label=r"maximizing $\ell$")
    ax_lossuv_zoom.plot(iterations_indexes[i+settings.v_max_iterations-1:i+settings.v_max_iterations+settings.u_max_iterations+1], lossuv_history[i+settings.v_max_iterations-1:i+settings.v_max_iterations+settings.u_max_iterations+1], linewidth=0.8,  color="blue", label=r"minimizing $\ell$")
    i=i+settings.v_max_iterations+settings.u_max_iterations
ax_lossuv_zoom.set_yscale('symlog', linthresh=10**(-5))
ax_lossuv_zoom.legend([r"maximizing $\ell$", r"minimizing $\ell$"])

ax_lossuv = fig.add_subplot(gs[1, 0])
ax_lossuv.set_yscale('symlog', linthresh=10**(-5))
ax_lossuv.plot(iterations_indexes, lossuv_history, linewidth=1.2, marker="None", color="gray", label=r"$\ell$")
ax_lossuv.plot(iterations_indexes, optimal_loss_uv*np.ones_like(iterations_indexes), linestyle="dashed", linewidth=1.2, marker="None", color="red", label=r"optimal $\ell$")
ax_lossuv.legend()

ax_lossuv_zoom_final = fig.add_subplot(gs[2, 0])
#ax_lossT_zoom_final.set_title(r'$\ell_T = \frac{1}{2}||T_{net}u_{net}||^2_V - l(T_{net}u_{net})$ evolution')
ax_lossuv_zoom_final.set_xlabel("iteration")
ax_lossuv_zoom_final.plot(iterations_indexes[-int(0.5*settings.num_iterations):], lossuv_history[-int(0.5*settings.num_iterations):], linewidth=1.2, marker="None", color="gray", label=r"$\ell$")
ax_lossuv_zoom_final.plot(iterations_indexes[-int(0.5*settings.num_iterations):], optimal_loss_uv*np.ones_like(iterations_indexes[-int(0.5*settings.num_iterations):]), linestyle="dashed", linewidth=1.2, marker="None", color="red", label=r"optimal $\ell$")
ax_lossuv_zoom_final.legend()

fig.show()
pdf.savefig(fig)

pdf.close()