def investigate_course_computations(self):
    """Compare different course computations over the Reel-In segment in one plot."""

    # -----------------------------
    # Cartesian course from dx, dy
    # -----------------------------
    self.course_cart = np.arctan2(
        self.dy_cyc[self.RI_start_idx:self.RI_end_idx],
        self.dx_cyc[self.RI_start_idx:self.RI_end_idx]
    )
    self.course_cart = np.mod(self.course_cart, 2 * np.pi)

    # -----------------------------
    # Spherical course from az, el
    # -----------------------------
    self.az_dot = np.gradient(self.az_RI)
    self.el_dot = np.gradient(self.el_RI)

    self.course_sph = -np.arctan2(self.az_dot * np.cos(self.el_RI), self.el_dot) + 2*np.pi
    self.course_sph = np.mod(self.course_sph, 2 * np.pi)

    # -----------------------------
    # Reference course from CSV
    # -----------------------------

    self.course_ref = np.mod(self.course_RI, 2 * np.pi)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(self.course_cart, label="Cartesian course", color="C0")
    ax.plot(self.course_sph, label="Spherical course", color="C1")
    ax.plot(self.course_ref, label="Reference (CSV)", color="C2")

    ax.set_ylabel("Course [rad]")
    ax.set_xlabel("Sample index")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Comparison of Course Computations over Reel-In Segment")

    plt.show()