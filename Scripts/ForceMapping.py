import numpy as np
class forceMapping:
    def __init__(self):
        self.fs2_init()
        self.fs3_init()
        self.fs4_init()
        self.fs6_init()
      
    def fs2_init(self):
        
        self.fs2_scale = 0.05856057
        
        self.fs2_base_ax = np.array([ 0.79,  0.61, -0.08])
        
        self.fs2_xy_ax = np.array([[-0.53,  0.6 , -0.6 ],
                                  [-0.44,  0.64,  0.63]])


    def fs3_init(self):

        self.fs3_scale = 0.07494279
        
        self.fs3_base_ax = np.array([ -0.77, -0.62,  0.14])
        
        self.fs3_xy_ax = np.array([[-0.59,  0.62, -0.51],
                                  [-0.46,  0.7 ,  0.54]])

    def fs4_init(self):

        self.fs4_scale = 0.06141585
        
        self.fs4_base_ax = np.array([ 0.79241273,  0.5969412 , -0.12547214])
        
        self.fs4_xy_ax = np.array([[-0.53244957,  0.57653486, -0.61975834],
                                   [-0.39563759,  0.65952432,  0.63909784]])
        
    def fs6_init(self):
   
        self.fs6_scale = 0.07347168
        
        self.fs6_base_ax = np.array([ 0.80872984,  0.57528534, -0.12248601])
        
        self.fs6_xy_ax = np.array([[-0.51985483,  0.62559465, -0.58170637],
                                   [-0.31499309,  0.60646952,  0.73005073]])

    
    


    def get_fs2(self,v):
        tmp = self.get_the_force(v, self.fs2_base_ax, self.fs2_scale,self.fs2_xy_ax)
        return tmp
    
    def get_fs3(self,v):
        tmp = self.get_the_force(v, self.fs3_base_ax, self.fs3_scale,self.fs3_xy_ax)
        return tmp

    def get_fs4(self,v):
        tmp = self.get_the_force(v, self.fs4_base_ax,self.fs4_scale,self.fs4_xy_ax)
        return tmp

    def get_fs6(self,v):
        tmp = self.get_the_force(v, self.fs6_base_ax, self.fs6_scale,self.fs6_xy_ax)
        return tmp

    def compute_expression(self, param, x):
        return param[0] * x**param[1] + param[2]

    
    def get_the_force(self,v, v_normal, force_sacle, optimal_xy_proj):
        # step3 get the total force
        total_force = np.sum(np.abs(v))*force_sacle
        # set the threshold for force_scale
        if total_force > 6.55:
            total_force = 6.55
        # step4 find the combination of the x and y
        proj_xy = self.project_to_plane(v_normal,v)
        proj_v = self.solve_linear_combination(optimal_xy_proj,proj_xy)
        # final get the force by direction and scaling
        res = self.get_the_force_inner(proj_v, total_force)
        return np.round(res,2)

    def get_the_force_inner(self, proj_v, total_force):
        force = np.zeros(3)
        force[:2] = (proj_v / np.linalg.norm(proj_v))
        return np.round(force,3)  * total_force

    def project_to_plane(self, v1, v2):
        """
        Projects v2 onto the plane with normal vector v1.
        
        Parameters:
            v1 (numpy array): Normal vector of the plane.
            v2 (numpy array): Vector to be projected.
        
        Returns:
            numpy array: Projection of v2 onto the plane.
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # Compute the projection of v2 onto v1
        projection_v2_on_v1 = (np.dot(v2, v1) / np.dot(v1, v1)) * v1
        
        # Compute the projection of v2 onto the plane
        v2_proj_plane = v2 - projection_v2_on_v1
        
        return v2_proj_plane
        
    def solve_linear_combination(self, data, v3):
        """
        Solves for a, b in the equation: v3 = a * v1 + b * v2 using least squares.
        
        Parameters:
            v1 (numpy array): First basis vector.
            v2 (numpy array): Second basis vector.
            v3 (numpy array): Target vector.
        
        Returns:
            tuple: (a, b) coefficients
        """
        v1 = data[0,:]
        v2 = data[1,:]
      
        
        # Construct matrix A with v1 and v2 as columns
        A = np.column_stack((v1, v2))
        
        # Solve for a and b using least squares
        solution, _, _, _ = np.linalg.lstsq(A, v3, rcond=None)
        
        return np.round(np.array([solution[0], solution[1]]),2)
        
    def angle_between_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0  # or return np.nan if you want to handle it differently
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return np.degrees(angle)