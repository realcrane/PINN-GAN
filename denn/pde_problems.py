import numpy as np
import torch
from denn.problem import Problem
from denn.utils import diff
from denn.burgers.fft_burgers import fft_burgers

class PoissonEquation(Problem):
    def __init__(self, nx=32, ny=32, xmin=0, xmax=1, ymin=0, ymax=1, batch_size=100, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.batch_size = batch_size
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.hy = (ymax - ymin) / ny
        self.noise_xstd = self.hx / 4.0
        self.noise_ystd = self.hy / 4.0

        xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        ygrid = torch.linspace(ymin, ymax, ny, requires_grad=True)

        grid_x, grid_y = torch.meshgrid(xgrid, ygrid, indexing='ij')
        self.grid_x, self.grid_y = grid_x.reshape(-1,1), grid_y.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_y)

    def get_grid_sample(self, grid, resid, resid_delta):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        y_noisy = torch.normal(mean=self.grid_y, std=self.noise_ystd)
        return (x_noisy, y_noisy)

    def get_plot_grid(self, grid_samp, residuals, residuals_delta):
        return self.get_grid()

    def get_plot_dims(self):
        return {'x': self.nx, 'y': self.ny}

    def get_solution(self, x, y):
        sol = x * (1-x) * y * (1-y) * torch.exp(x - y)
        return sol

    def get_plot_solution(self, x, y):
        return self.get_solution(x, y)

    def _poisson_eqn(self, u, x, y):
        return diff(u, x, order=2) + diff(u, y, order=2) - 2*x * (y - 1) * (y - 2*x + x*y + 2) * torch.exp(x - y)

    def get_equation(self, u, x, y):
        adj = self.adjust(u, x, y)
        u_adj = adj['pred']
        return self._poisson_eqn(u_adj, x, y)

    def adjust(self, u, x, y):
        self.x_min_val = lambda y: 0
        self.x_max_val = lambda y: 0
        self.y_min_val = lambda x: 0
        self.y_max_val = lambda x: 0

        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        y_tilde = (y-self.ymin) / (self.ymax-self.ymin)

        Axy = (1-x_tilde)*self.x_min_val(y) + x_tilde*self.x_max_val(y) + \
              (1-y_tilde)*( self.y_min_val(x) - ((1-x_tilde)*self.y_min_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_min_val(self.xmax * torch.ones_like(x_tilde))) ) + \
                 y_tilde *( self.y_max_val(x) - ((1-x_tilde)*self.y_max_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_max_val(self.xmax * torch.ones_like(x_tilde))) )

        u_adj = Axy + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, y, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, y)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, y)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.cpu().detach())}
        return pred_dict, diff_dict

class Laplace1Equation(Problem):

    def __init__(self, nx=32, ny=32, xmin=0, xmax=1, ymin=0, ymax=1, batch_size=100, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.batch_size = batch_size
        self.pi = torch.tensor(np.pi)
        self.ex = torch.tensor(1)
        self.hx = (xmax - xmin) / nx
        self.hy = (ymax - ymin) / ny
        self.noise_xstd = self.hx / 4.0
        self.noise_ystd = self.hy / 4.0

        xgrid = torch.linspace(xmin, xmax, nx+1, requires_grad=True)
        ygrid = torch.linspace(ymin, ymax, ny+1, requires_grad=True)

        grid_x, grid_y = torch.meshgrid(xgrid, ygrid, indexing='ij')
        self.grid_x_mean = grid_x
        self.grid_t_mean = grid_y
        self.grid_x, self.grid_y = grid_x.reshape(-1,1), grid_y.reshape(-1,1)

        xgrid1 = torch.linspace(xmin+0.1, xmax-0.1, nx+1 , requires_grad=True)
        ygrid1 = torch.linspace(ymin+0.1, ymax-0.1, ny+1 , requires_grad=True)

        grid_x1, grid_y1 = torch.meshgrid(xgrid1, ygrid1, indexing='ij')
        self.grid_x1, self.grid_y1 = grid_x1.reshape(-1, 1), grid_y1.reshape(-1, 1)

    def get_grid_sample_ub(self):
        grid_x, grid_t = self.grid_x_mean[-1, :].reshape(-1, 1), self.grid_t_mean[0, :].reshape(-1, 1)
        return (grid_x, grid_t)

    def get_grid_sample_lb(self):
        grid_x, grid_t = self.grid_x_mean[0, :].reshape(-1, 1), self.grid_t_mean[0, :].reshape(-1, 1)
        return (grid_x, grid_t)

    def get_grid_sample_ub_all(self):
        grid_x, grid_t = self.grid_x_mean[-1, :].reshape(-1, 1), self.grid_t_mean[0, :].reshape(-1, 1)
        return torch.cat((grid_x, grid_t), dim=1)

    def get_grid_sample_ini(self):
        grid_x, grid_t = self.grid_x_mean[:, 0].reshape(-1, 1), self.grid_t_mean[:, 0].reshape(-1, 1)
        return (grid_x, grid_t)

    def get_grid_sample_final(self):
        grid_x, grid_t = self.grid_x_mean[:, 0].reshape(-1, 1), self.grid_t_mean[:, -1].reshape(-1, 1)
        return (grid_x, grid_t)

    def get_grid_sample(self, grid_samp, residuals, residuals_delta):
        return (self.grid_x, self.grid_y)

    def get_grid(self):
        return (self.grid_x1, self.grid_y1)

    def get_plot_grid(self, grid_samp, residuals, residuals_delta):
        return self.get_grid_sample(grid_samp, residuals, residuals_delta)

    def get_plot_dims(self):
        return {'x': self.nx+1, 'y': self.ny+1}

    def get_solution(self, x, y):
        sol = 1/2/torch.cosh(self.pi)*torch.sin(self.pi*x)*(torch.exp(self.pi*y)-torch.exp(-self.pi*y))
        return sol

    def get_plot_solution(self, x, y):
        return self.get_solution(x, y)

    def _poisson_eqn(self, u, x, y):
        return diff(u, x, order=2) + diff(u, y, order=2)

    def get_equation(self, u, x, y):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, y)
        u_adj = adj['pred']
        return self._poisson_eqn(u_adj, x, y)

    def get_equation_soft(self, u, x, y, model):
        """ return value of residuals of equation (i.e. LHS) """
        residual = self._poisson_eqn(u, x, y)
        #print('residual',residual.shape)
        adj = self.adjust(u, x, y)
        bound = self.get_boundary_soft(model)
        loss = torch.cat([residual , bound], dim=0)
        #print('loss',loss.shape)
        return loss

    def get_boundary_soft(self, model1):
        x_ub, y_ub = self.get_grid_sample_ub()
        x_lb, y_lb = self.get_grid_sample_lb()
        x_ini, y_ini = self.get_grid_sample_ini()
        x_final, y_final = self.get_grid_sample_final()

        grid_samp_ub = torch.cat((x_ub, y_ub), 1)
        grid_samp_lb = torch.cat((x_lb, y_lb), 1)
        grid_samp_ini = torch.cat((x_ini, y_ini), 1)
        grid_samp_final = torch.cat((x_final, y_final), 1)

        pred1_ub = model1(grid_samp_ub)
        pred1_lb = model1(grid_samp_lb)
        pred1_ini = model1(grid_samp_ini)
        pred1_final = model1(grid_samp_final)

        ub_loss = pred1_ub- torch.zeros_like(pred1_ub)
        lb_loss = pred1_lb- torch.zeros_like(pred1_lb)
        ini_loss = pred1_ini- torch.zeros_like(pred1_ini)

        xgrid = torch.linspace(0, 1, 32+1, requires_grad=True)
        self.y_max_val = lambda x: 1/2/torch.cosh(self.pi)*torch.sin(self.pi*x)*(torch.exp(self.pi)-torch.exp(-self.pi))
        top_loss = pred1_final - self.y_max_val(xgrid.reshape(-1, 1))
        v_cat = torch.cat([ub_loss, lb_loss, ini_loss, top_loss], dim=0)
        #print('v_cat',v_cat.shape)
        return v_cat

    def adjust(self, u, x, y):
        self.x_min_val = lambda y: 0
        self.x_max_val = lambda y: 0
        self.y_min_val = lambda x: 0
        self.y_max_val = lambda x: 1/2/torch.cosh(self.pi)*torch.sin(self.pi*x)*(torch.exp(self.pi)-torch.exp(-self.pi))


        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        y_tilde = (y-self.ymin) / (self.ymax-self.ymin)

        Axy = (1-x_tilde)*self.x_min_val(y) + x_tilde*self.x_max_val(y) + \
              (1-y_tilde)*( self.y_min_val(x) - ((1-x_tilde)*self.y_min_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_min_val(self.xmax * torch.ones_like(x_tilde))) ) + \
                 y_tilde *( self.y_max_val(x) - ((1-x_tilde)*self.y_max_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_max_val(self.xmax * torch.ones_like(x_tilde))) )

        u_adj = Axy + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, y, sol):
        adj = self.adjust(pred, x, y)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, y)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach().cpu())}
        return pred_dict, diff_dict

    def get_plot_dicts_soft(self, pred, x, y, sol):
        pred_dict = {'$\hat{u}$': pred.detach()}
        resid = self.get_equation(pred, x, y)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach().cpu())}
        return pred_dict, diff_dict

class KleinEquation(Problem):

    def __init__(self, nx=32, nt=32, c=1, xmin=0, xmax=1, tmin=0, tmax=2, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.c = c
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0

        xgrid = torch.linspace(xmin, xmax, nx+1, requires_grad=True)
        tgrid = torch.linspace(tmin, tmax, nt+1, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

        xgrid1 = torch.linspace(xmin+0.1, xmax-0.1, nx+1 , requires_grad=True)
        tgrid1 = torch.linspace(tmin+0.1, tmax-0.1, nt+1 , requires_grad=True)

        grid_x1, grid_t1 = torch.meshgrid(xgrid1, tgrid1, indexing='ij')
        self.grid_x1, self.grid_t1 = grid_x1.reshape(-1, 1), grid_t1.reshape(-1, 1)

    def get_grid_sample(self,grid_samp, residuals, residuals_delta):
        return (self.grid_x, self.grid_t)

    def get_grid(self):
        return (self.grid_x1, self.grid_t1)

    def get_plot_grid(self,grid_samp, residuals, residuals_delta):
        return self.get_grid_sample(grid_samp, residuals, residuals_delta)

    def get_plot_dims(self):
        return {'x': self.nx+1, 't': self.nt+1}

    def get_solution(self, x, t):
        sol = -0.2*torch.sin(self.pi * x)*torch.cos(self.pi * t)
        return sol

    def get_plot_solution(self, x, t):
        return self.get_solution(x, t)

    def _Klein_eqn(self, u, x, t):
        return diff(u, t, order=2) - (self.c**2)*diff(u, x, order=2) -4*u - 0.8*torch.sin(torch.pi*x)*torch.cos(torch.pi*t)

    def get_equation(self, u, x, t):
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._Klein_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        x_tilde = (x - self.xmin) / (self.xmax - self.xmin)
        t_tilde = (t - self.tmin) / (self.tmax - self.tmin)
        Axy = -0.2 * torch.sin(self.pi * x)
        u_adj = Axy + x_tilde * (1 - x_tilde) * (1 - torch.exp(-t_tilde ** 2)) * u
        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach().cpu())}
        return pred_dict, diff_dict

class Laplace2Equation(Problem):

    def __init__(self, nx=64, nt=64, xmin=0, xmax=1, ymin=0, ymax=1, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.nt = nt
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (ymax - ymin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0

        xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        tgrid = torch.linspace(ymin, ymax, nt, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self, grid, resid, resid_delta):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_plot_grid(self, grid, resid, resid_delta):
        return self.get_grid()

    def get_plot_dims(self):
        return {'x': self.nx, 't': self.nt}

    def get_solution(self, x, y):
        sol = 1/2/torch.cosh(self.pi) *torch.sin(self.pi*x) * \
               (torch.exp(self.pi*(y-1))+torch.exp(self.pi*(1-y)))+ \
               1/4/torch.cosh(self.pi)* torch.cos(self.pi * x+self.pi/2)* \
              (torch.exp(self.pi*y)-torch.exp(-self.pi*y))
        return sol

    def get_plot_solution(self, x, y):
        return self.get_solution(x, y)

    def _laplace_eqn(self, u, x, y):
        return diff(u, x, order=2) + diff(u, y, order=2)

    def get_equation(self, u, x, y):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, y)
        u_adj = adj['pred']
        return self._laplace_eqn(u_adj, x, y)

    def adjust(self, u, x, y):
        self.x_min_val = lambda y: 0
        self.x_max_val = lambda y: 0
        self.y_min_val = lambda x: torch.sin(self.pi * x)
        self.y_max_val = lambda x: 1/torch.cosh(self.pi)*torch.sin(self.pi*x)+1/2*torch.tanh(self.pi)*torch.cos(self.pi*x + self.pi/2)

        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        y_tilde = (y-self.ymin) / (self.ymax-self.ymin)

        Axy = (1-y_tilde)* self.y_min_val(x) + y_tilde *( self.y_max_val(x) - \
                        ((1-x_tilde)*self.y_max_val(self.xmin * torch.ones_like(x_tilde))+ x_tilde * \
                         self.y_max_val(self.xmax * torch.ones_like(x_tilde))) )

        u_adj = Axy + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach(), '$u$': sol.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.cpu().detach())}
        return pred_dict, diff_dict

class BurgersViscous(Problem):

    def __init__(self, nx=64, nt=64, nu=0.001, xmin=-5, xmax=5, tmin=0, tmax=2.5,
        xmin_p=-5, xmax_p=5, tmin_p=0, tmax_p=2.5, nx_p=1000, nt_p=100, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.nu = nu
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0
        self.xgrid = torch.linspace(xmin, xmax, nx+1, requires_grad=True)
        self.tgrid = torch.linspace(tmin, tmax, nt+1, requires_grad=True)
        grid_x, grid_t = torch.meshgrid(self.xgrid, self.tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

        self.xmin_p = xmin_p
        self.xmax_p = xmax_p
        self.tmin_p = tmin_p
        self.tmax_p = tmax_p
        self.nx_p = nx_p
        self.nt_p = nt_p
        self.xgrid_p = torch.linspace(xmin_p, xmax_p, self.nx_p, requires_grad=True)
        self.tgrid_p = torch.linspace(tmin_p, tmax_p, self.nt_p, requires_grad=True)
        grid_x_p, grid_t_p = torch.meshgrid(self.xgrid_p, self.tgrid_p, indexing='ij')
        self.grid_x_p, self.grid_t_p = grid_x_p.reshape(-1,1), grid_t_p.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self, t=None, resid=None, resid_delta=None):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_plot_grid(self, grid_samp, residuals, residuals_delta):
        return (self.grid_x_p.float(), self.grid_t_p.float())

    def get_plot_dims(self):
        return {'x': self.nx_p, 't': self.nt_p}

    def get_solution(self, x, t):
        device, dtype = x.device, x.dtype
        x_np = np.linspace(self.xmin, self.xmax, (self.nx*8)+1)
        t_np = np.linspace(self.tmin, self.tmax, self.nt+1)
        sol = fft_burgers(x_np, t_np, self.nu)
        sol = sol[:,::8]
        sol = sol.T
        sol = torch.as_tensor(sol.reshape(-1,1), device=device, dtype=dtype)
        return sol

    def get_plot_solution(self, x, t):
        device, dtype = x.device, x.dtype
        x_np = self.xgrid_p.detach().cpu().numpy()
        t_np = self.tgrid_p.detach().cpu().numpy()
        sol = fft_burgers(x_np, t_np, self.nu)
        sol = sol.T
        sol = torch.as_tensor(sol.reshape(-1,1), device=device, dtype=dtype)
        return sol

    def _burgers_eqn(self, u, x, t):
        return diff(u, t, order=1) + u*diff(u, x, order=1) - self.nu*diff(u, x, order=2)

    def get_equation(self, u, x, t):
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._burgers_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)
        Axt = 1/torch.cosh(x)

        u_adj = Axt + x_tilde*(1-x_tilde)*(1 - torch.exp(-t_tilde))*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach(), '$u$': sol.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach().cpu())}
        return pred_dict, diff_dict


class ReactionEquation(Problem):
    def __init__(self, nx=32, nt=32, c=1, xmin=0, xmax=1, tmin=0, tmax=5, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.c = c
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0

        xgrid = torch.linspace(xmin, xmax, nx+1, requires_grad=True)
        tgrid = torch.linspace(tmin, tmax, nt+1, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

        xgrid1 = torch.linspace(xmin+0.1, xmax-0.1, nx+1 , requires_grad=True)
        tgrid1 = torch.linspace(tmin+0.1, tmax-0.1, nt+1 , requires_grad=True)

        grid_x1, grid_t1 = torch.meshgrid(xgrid1, tgrid1, indexing='ij')
        self.grid_x1, self.grid_t1 = grid_x1.reshape(-1, 1), grid_t1.reshape(-1, 1)

    def get_grid_sample(self,grid_samp, residuals, residuals_delta):
        return (self.grid_x, self.grid_t)

    def get_grid(self):
        return (self.grid_x1, self.grid_t1)

    def get_plot_grid(self,grid_samp, residuals, residuals_delta):
        return self.get_grid_sample(grid_samp, residuals, residuals_delta)

    def get_plot_dims(self):
        return {'x': self.nx+1, 't': self.nt+1}

    def get_solution(self, x, t):
        sol =t/5*x*(1-x)
        return sol

    def get_plot_solution(self, x, t):
        return self.get_solution(x, t)

    def _Reaction_eqn(self, u, x, t):
        return diff(u, t, order=1) - diff(u, x, order=2) -10*u - 2/5*t + (2*t-1/5)*x*(1-x)

    def get_equation(self, u, x, t):
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._Reaction_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)

        u_adj = x_tilde*(1-x_tilde)*t_tilde*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach().cpu())}
        return pred_dict, diff_dict
