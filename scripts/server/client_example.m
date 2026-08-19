% Minimal MATLAB client for the AWETrim reelout optimization server.
% Uses only built-in webread/webwrite (no toolboxes).
%
% Units: angles rad, rates rad/s, lengths m, speeds m/s, tension N, time s.
% Frame: azimuth = 0 points downwind, elevation from the ground plane.
% NOTE: table.t restarts at 0 for every refreshed trajectory and the pattern
% is periodic in s — when swapping to a refreshed path mid-simulation, align
% on s (or azimuth/elevation), not on t.

BASE = "http://127.0.0.1:8000";   % or http://<server-ip>:8000
json = weboptions("MediaType", "application/json", "Timeout", 30);

%% 1. Check the server is up
disp(webread(BASE + "/health"))

%% 2. Initialize once (does not solve yet). Only "inflow_conditions" is
%  required; all tunable fields shown with their defaults — see README.md.
%  5.2 m/s at 6 m height, wind from the west, logarithmic profile
%  (profile_law 2, ~8 m/s at 100 m with this roughness).
init.inflow_conditions = struct("wind_speed", 5.2, ...
                                "wind_direction", 270.0, ...
                                "profile_law", 2, ...
                                "z0", 0.03);
init.length = 200.0;                       % initial tether length r0 [m]
init.initial_guess = struct( ...
    "curve_type", "lissajous", ...         % figure-8; or "helix" for circles
    "az_amp0",   0.3, ...                  % figure half-width [rad]
    "beta0",     0.35, ...                 % mean elevation [rad]
    "beta_amp0", 0.12, ...                 % figure half-height [rad]
    "downloops", true, ...                 % turn downward through the loops
    "M",         10);                      % B-spline control points
init.n_points = 100;                       % trajectory table rows
init.input_depower = 1.6;                  % depower setting
init.reg_weight = 1.0;                     % regularization weight
init.detect_simple_bounds = true;          % solver flag
webwrite(BASE + "/init", init, json);

%% 3. First optimization (cold start, ~10 s)
traj = optimize(BASE, json, struct());
fprintf("step %d: %.2f kW average power\n", ...
        traj.step_index, traj.metrics.avg_power_W / 1e3);

% The flight path as plain vectors (100 points):
t      = traj.table.t;               % time along the path [s]
az     = traj.table.azimuth;         % [rad]
el     = traj.table.elevation;       % [rad]
az_dot = traj.table.azimuth_dot;     % [rad/s]  feedforward rates
el_dot = traj.table.elevation_dot;   % [rad/s]
r      = traj.table.distance_radial; % tether length [m]
r_dot  = traj.table.speed_radial;    % reel-out speed [m/s]

% 3D view of the path:
x = r .* cos(el) .* cos(az);  y = r .* cos(el) .* sin(az);  z = r .* sin(el);
figure; plot3(x, y, z, "LineWidth", 1.5); grid on; axis equal
xlabel("x downwind [m]"); ylabel("y [m]"); zlabel("z [m]")
title("Optimized reelout pattern")

%% 4. Later, from your simulation loop: refresh with current conditions.
%  A measured profile is sent as a CUSTOM_* law: the heights/speeds samples
%  are fitted (here: profile_law 6 = log law + Gaussian jet, >= 5 samples).
request.inflow_conditions = struct("wind_speed", 8.4, ...
                                   "wind_direction", 265.0, ...
                                   "profile_law", 6, ...
                                   "heights", [10.0 50.0 100.0 200.0 300.0], ...
                                   "speeds",  [5.5 7.4 8.0 9.3 8.6]);
request.length = 220.0;              % current tether length in your sim
traj = optimize(BASE, json, request);
fprintf("step %d: re-anchored at r0 = %.0f m\n", ...
        traj.step_index, traj.spline.r0);

%% ------------------------------------------------------------------------
function traj = optimize(BASE, json, request)
% One re-optimization: POST /step, poll /status, GET /trajectory.
    request.wait = false;                       % 202 + polling, not blocking
    webwrite(BASE + "/step", request, json);   % returns immediately (202)
    while true                                  % poll until finished
        status = webread(BASE + "/status");
        if status.state == "converged", break, end
        if status.state == "failed"
            error("solve failed: %s (previous trajectory still available)", ...
                  status.last_error);
        end
        pause(1);
    end
    traj = webread(BASE + "/trajectory");
end
