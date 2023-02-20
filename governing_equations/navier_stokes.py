from IPython.display import display, Markdown


class NavierStokes:
    def __init__(self):
        self.dudt, self.dudx, self.dudy, self.dudz = r'$\frac{\partial u}{\partial t}$', r'$u \frac{\partial u}{\partial x}$', r'$v \frac{\partial u}{\partial y}$', r'$w \frac{\partial u}{\partial z}$'
        self.dpdx, self.gx = r'$-\frac{1}{\rho} \frac{\partial p}{\partial x}$', r'$g_x$'
        self.dduddx, self.dduddy, self.dduddz = r'$\frac{1}{\nu} \frac{\partial^2 u}{\partial x^2}$', r'$\frac{1}{\nu} \frac{\partial^2 u}{\partial y^2}$', r'$\frac{1}{\nu} \frac{\partial^2 u}{\partial z^2}$'

        self.dvdt, self.dvdx, self.dvdy, self.dvdz = r'$\frac{\partial v}{\partial t}$', r'$u \frac{\partial v}{\partial x}$', r'$v \frac{\partial v}{\partial y}$', r'$w \frac{\partial v}{\partial z}$'
        self.dpdy, self.gy = r'$-\frac{1}{\rho} \frac{\partial p}{\partial y}$', r'$g_y$'
        self.ddvddx, self.ddvddy, self.ddvddz = r'$\frac{1}{\nu} \frac{\partial^2 v}{\partial x^2}$', r'$\frac{1}{\nu} \frac{\partial^2 v}{\partial y^2}$', r'$\frac{1}{\nu} \frac{\partial^2 v}{\partial z^2}$'

        self.dwdt, self.dwdx, self.dwdy, self.dwdz = r'$\frac{\partial w}{\partial t}$', r'$u \frac{\partial w}{\partial x}$', r'$v \frac{\partial w}{\partial y}$', r'$w \frac{\partial w}{\partial z}$'
        self.dpdz, self.gz = r'$-\frac{1}{\rho} \frac{\partial p}{\partial z}$', r'$g_z$'
        self.ddwddx, self.ddwddy, self.ddwddz = r'$\frac{1}{\nu} \frac{\partial^2 w}{\partial x^2}$', r'$\frac{1}{\nu} \frac{\partial^2 w}{\partial y^2}$', r'$\frac{1}{\nu} \frac{\partial^2 w}{\partial z^2}$'

        self.assum = {self.gx: ['no gravity', 'gravity in y', 'gravity in z'],
                      self.dudt: ['u=0', 'steady', 'steady flow'],
                      self.dudx: ['u=0'],
                      self.dudy: ['u=0', 'v=0'],
                      self.dudz: ['u=0', 'w=0', '2D'],
                      self.dpdx: ['constant pressure in x', 'constant pressure', 'external flow'],
                      self.dduddx: ['u=0', 'inviscid'],
                      self.dduddy: ['u=0', 'inviscid'],
                      self.dduddz: ['u=0', 'inviscid', '2D'],

                      self.gy: ['no gravity', 'gravity in x', 'gravity in z'],
                      self.dvdt: ['v=0', 'steady', 'steady flow'],
                      self.dvdx: ['v=0', 'u=0'],
                      self.dvdy: ['v=0'],
                      self.dvdz: ['v=0', 'w=0', '2D'],
                      self.dpdy: ['constant pressure in y', 'constant pressure'],
                      self.ddvddx: ['v=0', 'inviscid'],
                      self.ddvddy: ['v=0', 'inviscid'],
                      self.ddvddz: ['v=0', 'inviscid', '2D'],

                      self.gz: ['no gravity', 'gravity in x', 'gravity in y', '2D'],
                      self.dwdt: ['w=0''steady', 'steady flow', '2D'],
                      self.dwdx: ['w=0', 'u=0', '2D'],
                      self.dwdy: ['w=0', 'v=0', '2D'],
                      self.dwdz: ['w=0', '2D'],
                      self.dpdz: ['constant pressure in z', 'constant pressure', '2D'],
                      self.ddwddx: ['w=0', 'inviscid', '2D'],
                      self.ddwddy: ['w=0', 'inviscid', '2D'],
                      self.ddwddz: ['w=0', 'inviscid', '2D']}

    def check(self, term, assumptions):
        for assumption in assumptions:
            if assumption in self.assum[term]:
                return False
        return True

    def simplify_naiver(self, assumptions):
        eqn_x, eqn_y, eqn_z = '', '', ''

        ## X-Direction ##
        for term in [self.dudt, self.dudx, self.dudy, self.dudz]:
            eqn_x += (term + ' $+$ ') if self.check(term, assumptions) else ''

        eqn_x = eqn_x.strip(' $+$ ')
        eqn_x = (f'${eqn_x}$' if eqn_x else ' $0$ ') + ' $=$ '

        for term in [self.dpdx, self.gx, self.dduddx, self.dduddy, self.dduddz]:
            eqn_x += (term + ' $+$ ') if self.check(term, assumptions) else ''

        eqn_x = eqn_x.strip(' $+$ ')
        eqn_x = f'${eqn_x}$' if f'${eqn_x}$' != '$0$  $=$' else '$0$'

        ## Y-Direction ##
        for term in [self.dvdt, self.dvdx, self.dvdy, self.dvdz]:
            eqn_y += (term + ' $+$ ') if self.check(term, assumptions) else ''

        eqn_y = eqn_y.strip(' $+$ ')
        eqn_y = (f'${eqn_y}$' if eqn_y else ' $0$ ') + ' $=$ '

        for term in [self.dpdy, self.gy, self.ddvddx, self.ddvddy, self.ddvddz]:
            eqn_y += (term + ' $+$ ') if self.check(term, assumptions) else ''

        eqn_y = eqn_y.strip(' $+$ ')
        eqn_y = f'${eqn_y}$' if f'${eqn_y}$' != '$0$  $=$' else '$0$'

        ## Z-Direction ##
        for term in [self.dwdt, self.dwdx, self.dwdy, self.dwdz]:
            eqn_z += (term + ' $+$ ') if self.check(term, assumptions) else ''

        eqn_z = eqn_z.strip(' $+$ ')
        eqn_z = (f'${eqn_z}$' if eqn_z else ' $0$ ') + ' $=$ '

        for term in [self.dpdz, self.gz, self.ddwddx, self.ddwddy, self.ddwddz]:
            eqn_z += (term + ' $+$ ') if self.check(term, assumptions) else ''

        eqn_z = eqn_z.strip(' $+$ ')
        eqn_z = f'${eqn_z}$' if f'${eqn_z}$' != '$0$  $=$' else '$0$'

        return eqn_x, eqn_y, eqn_z


if __name__ == "__main__":
    import sympy as sp
    from sympy import init_printing
    init_printing()

    assumptions = ['2D', 'no gravity', 'steady']
    gov = NavierStokes()
    eqn_x, eqn_y, eqn_z = gov.simplify_naiver(assumptions)
    sp.pprint(Markdown(eqn_x))
    # %%
    display(Markdown(f'The `Navier-Stokes` equations are:'))
    display(Markdown(f'x-direction: {eqn_x}'))
    display(Markdown(f'x-direction: {eqn_y}'))
    display(Markdown(f'x-direction: {eqn_z}'))
    # %% [markdown]
