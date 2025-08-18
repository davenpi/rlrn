use rand::Rng;

pub struct CartPole {
    // Environment parameters (CartPole-v1 standard)
    gravity: f64,
    #[allow(dead_code)]
    mass_cart: f64,
    mass_pole: f64,
    total_mass: f64,
    length: f64, // Half the pole's length
    pole_mass_length: f64,
    force_mag: f64,
    tau: f64, // Time step
    
    // Thresholds for termination
    theta_threshold_radians: f64,
    x_threshold: f64,
    
    // Current state: [x, x_dot, theta, theta_dot]
    state: [f64; 4],
    steps_beyond_terminated: Option<i32>,
    step_count: i32,
}

impl CartPole {
    pub fn new() -> Self {
        CartPole {
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            total_mass: 1.1, // mass_cart + mass_pole
            length: 0.5, // Half the pole's length
            pole_mass_length: 0.05, // mass_pole * length
            force_mag: 10.0,
            tau: 0.02, // 20ms time step
            
            theta_threshold_radians: 12.0 * 2.0 * std::f64::consts::PI / 360.0, // 12 degrees
            x_threshold: 2.4,
            
            state: [0.0; 4],
            steps_beyond_terminated: None,
            step_count: 0,
        }
    }
    
    pub fn reset(&mut self) -> [f64; 4] {
        let mut rng = rand::thread_rng();
        
        // Initialize state with small random values
        self.state = [
            rng.gen_range(-0.05..0.05), // x
            rng.gen_range(-0.05..0.05), // x_dot
            rng.gen_range(-0.05..0.05), // theta
            rng.gen_range(-0.05..0.05), // theta_dot
        ];
        
        self.steps_beyond_terminated = None;
        self.step_count = 0;
        
        self.state
    }
    
    pub fn step(&mut self, action: usize) -> ([f64; 4], f64, bool, bool) {
        assert!(action < 2, "Action must be 0 or 1");
        
        let force = if action == 1 { self.force_mag } else { -self.force_mag };
        
        let [x, x_dot, theta, theta_dot] = self.state;
        
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        
        // Physics calculations
        let temp = (force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp) / 
            (self.length * (4.0/3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass));
        let x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;
        
        // Update state using Euler integration
        let new_x = x + self.tau * x_dot;
        let new_x_dot = x_dot + self.tau * x_acc;
        let new_theta = theta + self.tau * theta_dot;
        let new_theta_dot = theta_dot + self.tau * theta_acc;
        
        self.state = [new_x, new_x_dot, new_theta, new_theta_dot];
        self.step_count += 1;
        
        // Check termination conditions
        let terminated = new_x.abs() > self.x_threshold || 
                        new_theta.abs() > self.theta_threshold_radians;
        
        // Check truncation (max episode length)
        let truncated = self.step_count >= 500;
        
        // Calculate reward
        let reward = if !terminated {
            1.0
        } else if self.steps_beyond_terminated.is_none() {
            self.steps_beyond_terminated = Some(0);
            1.0
        } else {
            self.steps_beyond_terminated = Some(self.steps_beyond_terminated.unwrap() + 1);
            0.0
        };
        
        (self.state, reward, terminated, truncated)
    }
    
    pub fn state_dim(&self) -> usize {
        4
    }
    
    pub fn action_dim(&self) -> usize {
        2
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}
