""" Leftover code """

# def fourier_curvature(s, L, A_coeffs, B_coeffs):
#     """
#     s: array of arc lengths
#     L: total curve length (period)
#     A_coeffs, B_coeffs: arrays of coefficients, length N
#     """
#     N = len(A_coeffs)
#     kappa = np.zeros_like(s)
#     for n in range(1, N+1):
#         kappa += A_coeffs[n-1] * np.cos(2 * np.pi * n * s / L)
#         kappa += B_coeffs[n-1] * np.sin(2 * np.pi * n * s / L)
#     return kappa

# def fourier_torsion(s, L, C_coeffs, D_coeffs):
#     N = len(C_coeffs)
#     tau = np.zeros_like(s)
#     for n in range(1, N+1):
#         tau += C_coeffs[n-1] * np.cos(2 * np.pi * n * s / L)
#         tau += D_coeffs[n-1] * np.sin(2 * np.pi * n * s / L)
#     return tau
# # Parameters
# L = 50
# n_points = 5000
# s = np.linspace(0, L, n_points)
# n_modes = 2  # number of Fourier modes

# # Initial coefficients for 2 modes per coordinate
# A_init = [0.0, 0.0]
# B_init = [0.0, 0.0]
# C_init = [0.0, 0.0]
# D_init = [0.0, 0.0]
# E_init = [0.0, 0.0]
# F_init = [0.0, 0.0]

# # Vectorized Fourier sum function
# def fourier_sum(s, L, cos_coeffs, sin_coeffs):
#     n = np.arange(1, len(cos_coeffs)+1)[:, np.newaxis]
#     s = s[np.newaxis, :]
#     return np.sum(cos_coeffs[:, np.newaxis]*np.cos(2*np.pi*n*s/L) +
#                   sin_coeffs[:, np.newaxis]*np.sin(2*np.pi*n*s/L), axis=0)

# # Initial curve
# x = fourier_sum(s, L, np.array(A_init), np.array(B_init))
# y = fourier_sum(s, L, np.array(C_init), np.array(D_init))
# z = fourier_sum(s, L, np.array(E_init), np.array(F_init))

# # Set up figure
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')
# line, = ax.plot(x, y, z, lw=2)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.subplots_adjust(bottom=0.23)  # more space for sliders

# # Slider axes positions
# axcolor = 'lightgoldenrodyellow'
# slider_axes = []
# slider_labels = ['A1','B1','A2','B2','C1','D1','C2','D2','E1','F1','E2','F2']
# positions = [
#     [0.10, 0.16, 0.35, 0.02], [0.10, 0.13, 0.35, 0.02],
#     [0.55, 0.16, 0.35, 0.02], [0.55, 0.13, 0.35, 0.02],
#     [0.10, 0.10, 0.35, 0.02], [0.10, 0.07, 0.35, 0.02],
#     [0.55, 0.10, 0.35, 0.02], [0.55, 0.07, 0.35, 0.02],
#     [0.10, 0.04, 0.35, 0.02], [0.10, 0.01, 0.35, 0.02],
#     [0.55, 0.04, 0.35, 0.02], [0.55, 0.01, 0.35, 0.02]
# ]

# for pos in positions:
#     slider_axes.append(plt.axes(pos, facecolor=axcolor))

# # Slider objects
# sliders = [
#     Slider(slider_axes[0], 'A1', -10, 10, valinit=A_init[0]),
#     Slider(slider_axes[1], 'B1', -10, 10, valinit=B_init[0]),
#     Slider(slider_axes[2], 'A2', -10, 10, valinit=A_init[1]),
#     Slider(slider_axes[3], 'B2', -10, 10, valinit=B_init[1]),
#     Slider(slider_axes[4], 'C1', -10, 10, valinit=C_init[0]),
#     Slider(slider_axes[5], 'D1', -10, 10, valinit=D_init[0]),
#     Slider(slider_axes[6], 'C2', -10, 10, valinit=C_init[1]),
#     Slider(slider_axes[7], 'D2', -10, 10, valinit=D_init[1]),
#     Slider(slider_axes[8], 'E1', -10, 10, valinit=E_init[0]),
#     Slider(slider_axes[9], 'F1', -10, 10, valinit=F_init[0]),
#     Slider(slider_axes[10], 'E2', -10, 10, valinit=E_init[1]),
#     Slider(slider_axes[11], 'F2', -10, 10, valinit=F_init[1])
# ]

# # Update function
# def update(val):
#     A = [sliders[0].val, sliders[2].val]
#     B = [sliders[1].val, sliders[3].val]
#     C = [sliders[4].val, sliders[6].val]
#     D = [sliders[5].val, sliders[7].val]
#     E = [sliders[8].val, sliders[10].val]
#     F = [sliders[9].val, sliders[11].val]
    
#     x = fourier_sum(s, L, np.array(A), np.array(B))
#     y = fourier_sum(s, L, np.array(C), np.array(D))
#     z = fourier_sum(s, L, np.array(E), np.array(F))
    
#     line.set_xdata(x)
#     line.set_ydata(y)
#     line.set_3d_properties(z)
    
#     ax.set_xlim(np.min(x), np.max(x))
#     ax.set_ylim(np.min(y), np.max(y))
#     ax.set_zlim(np.min(z), np.max(z))
#     fig.canvas.draw_idle()

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update)

# plt.show()