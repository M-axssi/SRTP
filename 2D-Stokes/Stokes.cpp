#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>

#include <AFEPack/AMGSolver.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/EasyMesh.h>

#include <lac/sparse_matrix.h>
#include <lac/sparsity_pattern.h>
#include <lac/sparse_ilu.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/solver_cg.h>
#include <lac/precondition.h>
#include <lac/sparse_mic.h>
#include <lac/sparse_decomposition.h>

#include <lac/trilinos_sparse_matrix.h>
#include <lac/trilinos_block_sparse_matrix.h>
#include <lac/trilinos_vector.h>
#include <lac/trilinos_block_vector.h>
#include <lac/trilinos_precondition.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/lac/solver_gmres.h>

double f(double x)
{
    return 1 - x * x * x * x;
}

int main()
{
    TemplateGeometry<2> rectangle_template_geometry;
    rectangle_template_geometry.readData("rectangle.tmp_geo");
    CoordTransform<2, 2> rectangle_coord_transform;
    rectangle_coord_transform.readData("rectangle.crd_trs");
    TemplateDOF<2> rectangle_template_dof(rectangle_template_geometry);
    rectangle_template_dof.readData("rectangle.1.tmp_dof");
    BasisFunctionAdmin<double, 2, 2> rectangle_basis_function(rectangle_template_dof);
    rectangle_basis_function.readData("rectangle.1.bas_fun");
    TemplateElement<double, 2, 2> template_element;
    template_element.reinit(rectangle_template_geometry, rectangle_template_dof,
                            rectangle_coord_transform, rectangle_basis_function);
    /// 取了 2 次代数精度。
    const QuadratureInfo<2> &quad_info = template_element.findQuadratureInfo(4);
    int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<AFEPack::Point<2>> q_point = quad_info.quadraturePoint();
    int n_element_dof = template_element.n_dof();
    int n_bas = rectangle_basis_function.size();

    /// 设置边界
    double x0 = -1.0;
    double y0 = -1.0;
    double x1 = 1.0;
    double y1 = 1.0;
    /// 设置剖分断数和节点总数
    const int n = 20;
    const int dim1 = (2 * n - 1) * (2 * n - 1);
    const int dim = 2 * dim1 + (n + 1) * (n + 1);
    int index_map[2 * n + 2][2 * n + 2];
    int index = 0;

    static double A[dim + 1][dim + 1];
    double ux[2 * n + 2][2 * n + 2], uy[2 * n + 2][2 * n + 2];

    for (int i = 1; i <= 2 * n + 1; ++i)
        for (int j = 1; j <= 2 * n + 1; ++j)
            if (i == 1 || j == 1 || i == 2 * n + 1 || j == 2 * n + 1)
            {
                index_map[i][j] = -1;
                if (i == 1)
                {
                    uy[i][j] = 0;
                    ux[i][j] = f((double)(j - 1 - n) / n);
                    index_map[i][j] = -2;
                }
                else
                {
                    ux[i][j] = 0;
                    uy[i][j] = 0;
                }
            }
            else
            {
                index_map[i][j] = index;
                ++index;
            }

    int arr[4][4];
    std::vector<AFEPack::Point<2>> gv(4);
    std::vector<AFEPack::Point<2>> lv(4);
    std::vector<AFEPack::Point<2>> pgv(4);
    arr[0][0] = -1.0;
    arr[0][1] = -1.0;
    arr[1][0] = 1.0;
    arr[1][1] = -1.0;
    arr[2][0] = 1.0;
    arr[2][1] = 1.0;
    arr[3][0] = -1.0;
    arr[3][1] = 1.0;

    std::vector<unsigned int> nozeroperow(dim);
    Vector<double> rhs(dim);
    static int b[dim][dim];

    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
        {
            std::vector<int> pindex;
            std::vector<int> vindex;
            std::vector<std::pair<int, int>> coord;
            pindex.push_back(2 * dim1 - 1 + (n + 1) * (i - 1) + j);
            pindex.push_back(2 * dim1 - 1 + (n + 1) * (i - 1) + j + 1);
            pindex.push_back(2 * dim1 - 1 + (n + 1) * i + j);
            pindex.push_back(2 * dim1 - 1 + (n + 1) * i + j + 1);
            for (int k = 1; k <= 3; ++k)
                for (int o = 1; o <= 3; ++o)
                {
                    vindex.push_back(index_map[2 * (i - 1) + k][2 * (j - 1) + o]);
                    coord.push_back(std::make_pair(k, o));
                }
            for (int x = 0; x <= 8; ++x)
                for (int y = 0; y <= 8; ++y)
                    if (vindex[x] >= 0)
                        if (abs(coord[y].first - coord[x].first) <= 1 && abs(coord[y].second - coord[x].second) <= 1 && vindex[y] >= 0)
                        {
                            b[vindex[x]][vindex[y]] = 1;
                            b[vindex[x] + dim1][vindex[y] + dim1] = 1;
                        }
            for (int x = 0; x <= 8; ++x)
                if (vindex[x] >= 0)
                    for (auto y : pindex)
                    {
                        b[y][vindex[x]] = 1;
                        b[y][vindex[x] + dim1] = 1;
                        b[vindex[x]][y] = 1;
                        b[vindex[x] + dim1][y] = 1;
                    }
            for (auto x : pindex)
                for (auto y : pindex)
                    b[x][y] = 1;
        }
    for (int i = 0; i <= dim - 1; ++i)
        for (int j = 0; j <= dim - 1; ++j)
            if (b[i][j])
                ++nozeroperow[i];

    SparsityPattern sp_stiff_matrix(dim, nozeroperow);

    for (int i = 0; i <= dim - 1; ++i)
        for (int j = 0; j <= dim - 1; ++j)
            if (b[i][j])
            {
                sp_stiff_matrix.add(i, j);
            }

    sp_stiff_matrix.compress();
    SparseMatrix<double> stiff_mat(sp_stiff_matrix);

    //构造[A O B_X^T]
    //         [O A B_Y^T]
    //        [B_X B_Y O]
    //和右边项
    for (int i = 1; i <= 2 * n; ++i)
        for (int j = 1; j <= 2 * n; ++j)
        {
            int rindex = (i + 1) / 2;
            int cindex = (j + 1) / 2;
            std::vector<int> pindex;
            double py00 = (double)(n - 2 * rindex) / n;
            double px00 = (double)(2 * cindex - n - 2) / n;
            pindex.push_back(2 * dim1 + rindex * (n + 1) + cindex - 1);
            double py10 = (double)(n - 2 * rindex) / n;
            double px10 = (double)(2 * cindex - n) / n;
            pindex.push_back(2 * dim1 + rindex * (n + 1) + cindex);
            double py11 = (double)(n - 2 * rindex + 2) / n;
            double px11 = (double)(2 * cindex - n) / n;
            pindex.push_back(2 * dim1 + (rindex - 1) * (n + 1) + cindex);
            double py01 = (double)(n - 2 * rindex + 2) / n;
            double px01 = (double)(2 * cindex - n - 2) / n;
            pindex.push_back(2 * dim1 + (rindex - 1) * (n + 1) + cindex - 1);

            pgv[0][0] = px00;
            pgv[0][1] = py00;
            pgv[1][0] = px10;
            pgv[1][1] = py10;
            pgv[2][0] = px11;
            pgv[2][1] = py11;
            pgv[3][0] = px01;
            pgv[3][1] = py01;

            std::vector<int> vindex;
            std::vector<std::pair<int, int>> coor;
            double y00 = (double)(n - i) / n;
            double x00 = (double)(j - 1 - n) / n;
            vindex.push_back(index_map[i + 1][j]);
            coor.push_back(std::make_pair(i + 1, j));
            double y10 = (double)(n - i) / n;
            double x10 = (double)(j - n) / n;
            vindex.push_back(index_map[i + 1][j + 1]);
            coor.push_back(std::make_pair(i + 1, j + 1));
            double y11 = (double)(n - i + 1) / n;
            double x11 = (double)(j - n) / n;
            vindex.push_back(index_map[i][j + 1]);
            coor.push_back(std::make_pair(i, j + 1));
            double y01 = (double)(n - i + 1) / n;
            double x01 = (double)(j - n - 1) / n;
            vindex.push_back(index_map[i][j]);
            coor.push_back(std::make_pair(i, j));

            gv[0][0] = x00;
            gv[0][1] = y00;
            gv[1][0] = x10;
            gv[1][1] = y10;
            gv[2][0] = x11;
            gv[2][1] = y11;
            gv[3][0] = x01;
            gv[3][1] = y01;
            lv[0][0] = arr[0][0];
            lv[0][1] = arr[0][1];
            lv[1][0] = arr[1][0];
            lv[1][1] = arr[1][1];
            lv[2][0] = arr[2][0];
            lv[2][1] = arr[2][1];
            lv[3][0] = arr[3][0];
            lv[3][1] = arr[3][1];

            auto point = rectangle_coord_transform.local_to_global(q_point, lv, gv);
            for (int l = 0; l < n_quadrature_point; l++)
            {
                double Jxy = quad_info.weight(l) * rectangle_coord_transform.local_to_global_jacobian(q_point[l], lv, gv);
                for (int base1 = 0; base1 < template_element.n_dof(); base1++)
                {
                    if (vindex[base1] == -1)
                        continue;
                    if (vindex[base1] >= 0)
                    {
                        for (int base2 = 0; base2 < template_element.n_dof(); base2++)
                            if (vindex[base2] >= 0)
                            {
                                stiff_mat.add(vindex[base1], vindex[base2], Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv)));
                                stiff_mat.add(vindex[base1] + dim1, vindex[base2] + dim1, Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv)));
                            }
                            else if (vindex[base2] == -2)
                            {
                                rhs(vindex[base1]) -= ux[coor[base2].first][coor[base2].second] * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv));
                                rhs(vindex[base1] + dim1) -= uy[coor[base2].first][coor[base2].second] * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv));
                            }
                    }
                    for (int base2 = 0; base2 < template_element.n_dof(); base2++)
                    {
                        if (vindex[base1] >= 0)
                        {
                            std::vector<double> T1(2);
                            std::vector<double> T2(2);
                            T1[0] = rectangle_basis_function[base2].value(point[l], pgv);
                            T1[1] = 0;
                            T2[0] = 0;
                            T2[1] = T1[0];

                            stiff_mat.add(vindex[base1], pindex[base2], -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T1));
                            stiff_mat.add(pindex[base2], vindex[base1], -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T1));
                            stiff_mat.add(vindex[base1] + dim1, pindex[base2], -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T2));
                            stiff_mat.add(pindex[base2], vindex[base1] + dim1, -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T2));
                        }
                        else if (vindex[base1] == -2)
                        {
                            std::vector<double> T1(2);
                            std::vector<double> T2(2);
                            T1[0] = 1;
                            T1[1] = 0;
                            T2[0] = 0;
                            T2[1] = 1;
                            double value = rectangle_basis_function[base2].value(point[l], pgv);
                            rhs(pindex[base2]) += ux[coor[base1].first][coor[base1].second] * value * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T1);
                            rhs(pindex[base2]) += uy[coor[base1].first][coor[base1].second] * value * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T2);
                        }
                    }
                }
            }
        }

    //构造稳定化矩阵C
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
        {
            double volume = 4.0 / n / n;
            std::vector<int> pindex;
            double px00 = (double)(2 * j - 2 - n) / n;
            double py00 = (double)(n - 2 * i) / n;
            pindex.push_back(2 * dim1 + i * (n + 1) + j - 1);
            double px10 = (double)(2 * j - n) / n;
            double py10 = (double)(n - 2 * i) / n;
            pindex.push_back(2 * dim1 + i * (n + 1) + j);
            double px11 = (double)(2 * j - n) / n;
            double py11 = (double)(n - 2 * i + 2) / n;
            pindex.push_back(2 * dim1 + (i - 1) * (n + 1) + j);
            double px01 = (double)(2 * j - n - 2) / n;
            double py01 = (double)(n - 2 * i + 2) / n;
            pindex.push_back(2 * dim1 + (i - 1) * (n + 1) + j - 1);

            pgv[0][0] = px00;
            pgv[0][1] = py00;
            pgv[1][0] = px10;
            pgv[1][1] = py10;
            pgv[2][0] = px11;
            pgv[2][1] = py11;
            pgv[3][0] = px01;
            pgv[3][1] = py01;
            lv[0][0] = arr[0][0];
            lv[0][1] = arr[0][1];
            lv[1][0] = arr[1][0];
            lv[1][1] = arr[1][1];
            lv[2][0] = arr[2][0];
            lv[2][1] = arr[2][1];
            lv[3][0] = arr[3][0];
            lv[3][1] = arr[3][1];

            auto point = rectangle_coord_transform.local_to_global(q_point, lv, pgv);
            for (int l = 0; l < n_quadrature_point; l++)
            {
                double Jxy = quad_info.weight(l) * rectangle_coord_transform.local_to_global_jacobian(q_point[l], lv, pgv);
                for (int base1 = 0; base1 < template_element.n_dof(); base1++)
                {
                    for (int base2 = 0; base2 < template_element.n_dof(); base2++)
                    {
                        stiff_mat.add(pindex[base1], pindex[base2], -Jxy * (rectangle_basis_function[base1].value(point[l], pgv) - 1.0 / 4) * (rectangle_basis_function[base2].value(point[l], pgv) - 1.0 / 4));
                    }
                }
            }
        }

    Vector<double> solution(dim);

    double tol = std::numeric_limits<double>::epsilon() * dim;

    SolverControl solver_control(1000000, tol, false);
    //  求解器的选择, Stokes系统是对称不定的, 不能采用cg, 如果使用GMRES, 效率比较低.
    SolverGMRES<Vector<double>> gmres(solver_control);
    gmres.solve(stiff_mat, solution, rhs, PreconditionIdentity());

    for (int i = 2; i <= 2 * n; ++i)
        for (int j = 2; j <= 2 * n; ++j)
        {
            ux[i][j] = solution[index_map[i][j]];
            uy[i][j] = solution[index_map[i][j] + dim1];
        }

    std::ofstream fs;
    fs.open("output.m");
    fs << "y=1:-2/" << n << ":-1;" << std::endl;
    fs << "x=-1:2/" << n << ":1;" << std::endl;
    fs << "[X,Y]=meshgrid(x,y);" << std::endl;
    fs << "U=[";
    int c = 2 * dim1;
    for (int j = 0; j < n + 1; j++)
    {
        for (int i = 0; i < n + 1; i++)
            fs << solution[i + c + n * j] << " , ";
        fs << ";" << std::endl;
        c++;
    }
    fs << "]" << std::endl;
    fs << "surf(X,Y,U);" << std::endl;
    fs << std::endl;
    fs << "y=1:-1/" << n << ":-1;" << std::endl;
    fs << "x=-1:1/" << n << ":1;" << std::endl;
    fs << "[X,Y]=meshgrid(x,y);";
    fs << "P=[";
    for (int i = 1; i <= 2 * n + 1; ++i)
    {
        for (int j = 1; j <= 2 * n + 1; ++j)
            fs << -1 + (j - 1) * (1.0 / n) << "," << 1 - (i - 1) * (1.0 / n) << "," << ux[i][j] << "," << uy[i][j] << ";" << std::endl;
    }
    // fs << "V=[";
    // for (int i = 1; i <= 2 * n + 1; ++i)
    // {
    //     for (int j = 1; j <= 2 * n + 1; ++j)
    //         fs << uy[i][j] << ",";
    //     fs << ";" << std::endl;
    // }
    // fs << "]" << std::endl;
    // fs << "figure" << std::endl;
    // fs << "quiver(X,Y,U,V);" << std::endl;
    // fs << "starty=-1:3/" << n << ":1" << std::endl;
    // fs << "startx=zeros(size(starty));" << std::endl;
    // fs << "streamline(X,Y,U,V,startx,starty);" << std::endl;
    fs.close();
}
