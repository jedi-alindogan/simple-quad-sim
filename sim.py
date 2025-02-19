import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter
from neural_fly import utils
from neural_fly import mlmodel
import torch

def quat_mult(q, p):
    # q * p
    # p,q = [w x y z]
    return np.array(
        [
            p[0] * q[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] + q[2] * p[3] - q[3] * p[2],
            q[2] * p[0] + q[0] * p[2] + q[3] * p[1] - q[1] * p[3],
            q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1],
        ]
    )
    
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_vectors(v_from, v_to):
    v_from = normalized(v_from)
    v_to = normalized(v_to)
    v_mid = normalized(v_from + v_to)
    q = np.array([np.dot(v_from, v_mid), *np.cross(v_from, v_mid)])
    return q

def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm

NO_STATES = 13
IDX_POS_X = 0
IDX_POS_Y = 1
IDX_POS_Z = 2
IDX_VEL_X = 3
IDX_VEL_Y = 4
IDX_VEL_Z = 5
IDX_QUAT_W = 6
IDX_QUAT_X = 7
IDX_QUAT_Y = 8
IDX_QUAT_Z = 9
IDX_OMEGA_X = 10
IDX_OMEGA_Y = 11
IDX_OMEGA_Z = 12


class Robot:
    
    '''
    frames:
        B - body frame
        I - inertial frame
    states:
        p_I - position of the robot in the inertial frame (state[0], state[1], state[2])
        v_I - velocity of the robot in the inertial frame (state[3], state[4], state[5])
        q - orientation of the robot (w=state[6], x=state[7], y=state[8], z=state[9])
        omega - angular velocity of the robot (state[10], state[11], state[12])
    inputs:
        omega_1, omega_2, omega_3, omega_4 - angular velocities of the motors
    '''
    def __init__(self,
                 recording=True, 
                 vehicle='quadsim', 
                 trajectory='figure8', 
                 method='PID', 
                 condition='nowind', 
                 count=0,
                 basis=None
                 ):
        self.recording = recording

        # Parameters 
        self.m = 1.0 # mass of the robot
        self.arm_length = 0.25 # length of the quadcopter arm (motor to center)
        self.height = 0.05 # height of the quadcopter
        self.body_frame = np.array([(self.arm_length, 0, 0, 1),
                                    (0, self.arm_length, 0, 1),
                                    (-self.arm_length, 0, 0, 1),
                                    (0, -self.arm_length, 0, 1),
                                    (0, 0, 0, 1),
                                    (0, 0, self.height, 1)])
        self.J = 0.025 * np.eye(3) # [kg m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.constant_thrust = 10e-4
        self.constant_drag = 10e-6
        self.pwm = np.array([0.0, 0.0, 0.0, 0.0])
        self.state = self.reset_state_and_input(np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.time = 0.0

        # Disturbance
        self.fa = np.zeros(3)

        # Adaptation Model
        self.basis = basis
        if self.basis is not None:
            self.Phi = self.compute_Phi()
            self.ahat = np.zeros(9)  # dim_a * dim_u
            self.s = np.zeros(3)

            # Gain Matrices
            self._lam = 1               # adaptation gain
            self._Lam = np.eye(3)       # s gain matrix
            self._Q   = 1e-4 * np.eye(9)       # not sure what this is ngl
            self._P   = 0 * np.eye(9)       # covariance-like matrix
            self._K   = np.eye(3)       # tracking error gain
            self._R   = np.eye(3)       # also not sure what this is
            
            # Simulate measurement noise of the force
            self.measurement_cov = np.eye(3) * (self.m * 0.001) ** 2
        
        # Data logging
        self.data_log = {field: [] for field in ['t', 'p', 'p_d', 'v', 'q', 'R', 'w', 'fa', 'pwm']}
        self.data_log['vehicle'] = vehicle
        self.data_log['trajectory'] = trajectory
        self.data_log['method'] = method
        self.data_log['condition'] = condition
        self.data_log['count'] = count
    
    def record_data(self):
        """
        Records data from the simulation into a format used by the Neural Fly ML Model.
        """
        t = self.time
        p = self.state[IDX_POS_X:IDX_POS_Z+1].copy()
        p_d = self.p_d_I.copy()
        v = self.state[IDX_VEL_X:IDX_VEL_Z+1].copy()
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1].copy()
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().tolist()
        w = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1].copy()
        fa = self.fa.copy()
        pwm = self.pwm.copy()

        self.data_log['t'].append(t)
        self.data_log['p'].append(p)
        self.data_log['p_d'].append(p_d)             # Desired position placeholder
        self.data_log['v'].append(v)
        self.data_log['q'].append(q)
        self.data_log['R'].append(R)
        self.data_log['w'].append(w)
        self.data_log['fa'].append(fa)               # Placeholder aerodynamic force
        self.data_log['pwm'].append(pwm)
    
    def save_simulation_data(self, folder):
        # Convert time-series data to NumPy arrays
        data_dict = {key: np.array(self.data_log[key]) for key in self.data_log if isinstance(self.data_log[key], list)}

        # Ensure metadata fields are strings
        data_dict['vehicle'] = str(self.data_log['vehicle'])
        data_dict['trajectory'] = str(self.data_log['trajectory'])
        data_dict['method'] = str(self.data_log['method'])
        data_dict['condition'] = str(self.data_log['condition'])
        data_dict['count'] = str(self.data_log['count'])

        utils.save_data([data_dict], folder)

    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        return state0

    def update(self, omegas_motor, dt):
        # Record the current data
        self.pwm = omegas_motor
        if self.recording:
            self.record_data()

        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        thrust = self.constant_thrust * np.sum(omegas_motor**2)
        f_b = np.array([0, 0, thrust])
        
        tau_x = self.constant_thrust * (omegas_motor[3]**2 - omegas_motor[1]**2) * 2 * self.arm_length
        tau_y = self.constant_thrust * (omegas_motor[2]**2 - omegas_motor[0]**2) * 2 * self.arm_length
        tau_z = self.constant_drag * (omegas_motor[0]**2 - omegas_motor[1]**2 + omegas_motor[2]**2 - omegas_motor[3]**2)
        tau_b = np.array([tau_x, tau_y, tau_z])

        v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81]) + self.fa / self.m
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I
        
        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.

        # Composite Adaptation Updates
        if self.basis is not None:
            # Measure the residual force
            noise = np.random.multivariate_normal(mean=np.zeros(3), cov=self.measurement_cov)
            y = self.fa + noise # Intoducing noise so it's like a sensor.

            # Compute derivatives
            _Rinv = np.linalg.inv(self._R)
            ahat_dot = - self._lam * self.ahat \
                - self._P @ (self.Phi).T @ _Rinv @ (self.Phi @ self.ahat - y) \
                + self._P @ (self.Phi).T @ self.s
            _P_dot = - 2 * self._lam * self._P \
                + self._Q - self._P @ (self.Phi).T @ _Rinv @ self.Phi @ self._P
            
            # Update
            self.ahat += ahat_dot * dt
            self._P += _P_dot * dt

        self.time += dt

    def compute_Phi(self):
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        x = np.hstack((v_I, q, self.pwm))
        x_tensor = torch.from_numpy(x).to(torch.double)
        phi = self.basis(x_tensor).detach().numpy()
        zero = np.zeros_like(phi)
        Phi = np.hstack(
                (np.vstack((phi, zero, zero)), 
                 np.vstack((zero, phi, zero)), 
                 np.vstack((zero, zero, phi))))
        return Phi

    def control(self, p_d_I, v_d_I = np.zeros(3), a_d_I = np.zeros(3)):
        self.p_d_I = p_d_I
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]

        if self.basis is not None:
            # Use composite adaptation
            self.Phi = self.compute_Phi()
            self.s = v_I - v_d_I - self._Lam @ (p_d_I - p_I)
            a_r = a_d_I + self._Lam @ (v_d_I - v_I)
            f = self.m * a_r + self.m * np.array([0, 0, 9.81]) - self._K @ self.s - self.Phi @ self.ahat
        else:
            # Use PD control
            k_p = 1.
            k_d = 10.0
            v_r = - k_p * (p_I - p_d_I)
            a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81])
            f = self.m * a
        
        f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f
        thrust = np.max([0, f_b[2]])
        
        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b) # + self.J @ omega_ref_dot
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))
        return omega_motor
    
DURATION = 20
PLAYBACK_SPEED = 3
CONTROL_FREQUENCY = 200 # Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

def get_pos_full_quadcopter(quad):
    """ position returns a 3 x 6 matrix 
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
    """
    origin = quad.state[IDX_POS_X:IDX_POS_Z+1]
    quat = quad.state[IDX_QUAT_W:IDX_QUAT_Z+1]
    rot = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()
    wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
    quadBodyFrame = quad.body_frame.T
    quadWorldFrame = wHb.dot(quadBodyFrame)
    pos_full_quad = quadWorldFrame[0:3]
    return pos_full_quad

def control_propellers(quad, trajectory='figure8', scale=2):
    t = quad.time
    T = 2.5
    r = 2*np.pi * t / T
    rdot = 2 * np.pi / T
    if trajectory == 'figure8':
        p_d_I = np.array([
            scale * np.cos(r/2), 
            scale * np.sin(r), 
            np.sin(r/2)
        ])
        
        v_d_I = np.array([
            scale * (-0.5) * np.sin(r/2) * rdot,
            scale * np.cos(r) * rdot,
            0.5 * np.cos(r/2) * rdot
        ])
        
        a_d_I = np.array([
            scale * (-0.25) * np.cos(r/2) * rdot**2,
            scale * -np.sin(r) * rdot**2,
            -0.25 * np.sin(r/2) * rdot**2
        ])

    elif trajectory == 'circle':
        p_d_I = np.array([
            scale * np.cos(r), 
            scale * np.sin(r), 
            np.sin(r/2)
        ])

        v_d_I = np.array([
            scale * -np.sin(r) * rdot,
            scale * np.cos(r) * rdot,
            0.5 * np.cos(r/2) * rdot
        ])
        
        a_d_I = np.array([
            scale * -np.cos(r) * rdot**2,
            scale * -np.sin(r) * rdot**2,
            -0.25 * np.sin(r/2) * rdot**2
        ])
        prop_thrusts = quad.control(p_d_I = np.array([scale * np.cos(r), scale * np.sin(r), np.sin(r/2)]))

    elif trajectory =='hover':
        p_d_I = np.array([2, 0, 0])
        v_d_I = np.zeros(3)
        a_d_I = np.zeros(3)
        prop_thrusts = quad.control(p_d_I)   # Hover Mode

    else:
        raise(ValueError, 'Trajectory not implemented.')
    
    prop_thrusts = quad.control(p_d_I, v_d_I, a_d_I)
    quad.update(prop_thrusts, dt)

def update_disturbance(quad, F0, alpha, beta):
    direction = np.array([np.cos(alpha) * np.sin(beta),
                          np.sin(alpha) * np.sin(beta),
                          np.cos(beta)])
    t = quad.time
    T = 10
    r = np.pi * t / T
    fa = (F0 * np.sin(r)**2) * direction
    noise = np.random.uniform(-1/30, 1/30)
    quad.fa = fa + noise

def generate_dataset(folder, trajectory='figure8', count=0, F0=0, alpha=0, beta=np.pi/2):
    quadcopter = Robot(recording=True, trajectory=trajectory, condition=F0, count=count)
    steps = int(DURATION / dt)

    # Generate a forward or backward trajectory
    scale = np.random.choice([-2,2])
    for _ in range(steps):
        update_disturbance(quadcopter, F0, alpha, beta)
        control_propellers(quadcopter, trajectory, scale)
    quadcopter.save_simulation_data(folder)

    print(f'Simulation {count} complete. Data saved.')

def simulate(basis=None):
    quadcopter = Robot(recording=False, basis=basis)
    def control_loop(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter)
        return get_pos_full_quadcopter(quadcopter)
    plotter = QuadPlotter()
    plotter.plot_animation(control_loop)

def generate_all_data():
    # Training sets
    folder = './data/training'
    F0s = [4, 8, 12, 16]
    alphas = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
    betas = [5 * np.pi/12, 7 * np.pi/12]
    combinations = [(f, a, b) for f in F0s for a in alphas for b in betas]
    combinations.append((0,0,0))

    for (count, combo) in enumerate(combinations):
        F0 = combo[0]
        alpha = combo[1]
        beta = combo[2]
        p = np.random.uniform(0,1)
        if p >= 0 and p < 0.4:
            generate_dataset(folder,'figure8', count, F0, alpha, beta)
        if p >= 0.4 and p < 0.8:
            generate_dataset(folder,'circle', count, F0, alpha, beta)
        if p >= 0.8:
            generate_dataset(folder,'hover', count, F0, alpha, beta)


    # Test sets
    folder = './data/testing'
    for count in range(10):
        F0 = np.random.choice([5,6,7,8,9,10,11,12,13,14,15,16], replace=False)
        alpha = np.random.uniform(0, np.pi/2)
        beta = np.random.uniform(5 * np.pi/12, 7 * np.pi/12)
        p = np.random.uniform(0,1)
        if p >= 0 and p < 0.4:
            generate_dataset(folder,'figure8', count, F0, alpha, beta)
        if p >= 0.4 and p < 0.8:
            generate_dataset(folder,'circle', count, F0, alpha, beta)
        if p >= 0.8:
            generate_dataset(folder,'hover', count, F0, alpha, beta)

def main():
    generate_all_data()
        # dim_a = 3
        # features = ['v', 'q', 'pwm']
        # dataset = 'neural-fly'
        # modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"

        # stopping_epoch = 200
        # folder = './hw2/simple-quad-sim/neural_fly/models/'
        # model = mlmodel.load_model(modelname = modelname + '-epoch-' + str(stopping_epoch), modelfolder=folder)
        # model.options['num_epochs'] = 200
        # basis = model.phi
        # simulate(basis=basis)

if __name__ == "__main__":
    main()