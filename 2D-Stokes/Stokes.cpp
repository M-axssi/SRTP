#include <iostream>
#include <fstream>
#include <limits>

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

double f(double x)
{
    return 1 - x * x * x * x;
}

int main()
{
    TemplateGeometry<2> rectangle_template_geometry;
    rectangle_template_geometry.readData("rectangle.tmp_geo");
    CoordTransform<2, 2> rectangle_coord_transform;
    rectangle_coord_transform.readDate("rectangle.crd_trs");
    TemplateDOF<2> rectangle_template_dof(rectangle_template_geometry);
    rectangle_template_dof.readData("rectangle.1.tmp_dof");
    BasisFunctionAdmin<double, 2, 2> rectangle_basis_function(rectangle_template_dof);
    rectangle_basis_function.readData("rectangle.1.bas_fun");
    TemplateElement<double, 2, 2> template_element;
    template_element.reinit(rectangle_template_geometry, rectangle_template_dof,
                            rectangle_coord_transform, rectangle_basis_function);

    double volume = template_element.volume();
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
    int n = 20;
    int dim1 = (2 * n - 1) * (2 * n - 1);
    int dim = 2 * dim1 + (n + 1) * (n + 1);
    int index_map[2 * n + 1, 2 * n + 1];
    int index = 0;

    //边界点插值
    double ux[2 * n + 1][2 * n + 1], uy[2 * n + 1][2 * n + 1];

    for (int i = 1; i <= 2 * n + 1; ++i)
        for (int j = 1; j <= 2 * n + 1; ++j)
        {
            if (i == 1 || j == 1 || i == 2 * n + 1 || j == 2 * n + 1)
            {
                index_map[i, j] = -1;
                if (i == 1)
                {
                    uy[i][j] = 0;
                    ux[i][j] = f((j - 1 - n) / n);
                    index_map[i, j] = -2;
                }
            }
            else
            {
                index_map[2 * n + 1, 2 * n + 1] = index;
                ++index;
            }
        }
    int arr[4][4];
    std::vector<AFEPack::Point<2>> gv(4);
    std::vector<AFEPack::Point<2>> lv(4);
    std::vector<AFEPack::Point<2>> pgv(4);
    std::vector<AFEPack::Point<2>> plv(4);
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
    {
        for (int i = 0; i <= 2 * n - 2; ++i)
            for (int j = 0; j <= 2 * n - 2; ++j)
            {
                int index = i * (2 * n - 1) + j;
                int index1 = i * (2 * n - 1) + j + dim1;
                if (i % 2 == 0)
                {
                    if (j % 2 == 0)
                    {
                        nozeroperow[index] = 13;
                        nozeroperow[index1] = 13;
                    }
                    else
                    {
                        nozeroperow[index] = 15;
                        nozeroperow[index1] = 15;
                    }
                }
                else
                {
                    if (j % 2 == 0)
                    {
                        nozeroperow[index] = 15;
                        nozeroperow[index1] = 15;
                    }
                    else
                    {
                        nozeroperow[index] = 18;
                        nozeroperow[index1] = 18;
                    }
                }
                if (i == 0 || i == 2 * n - 2 || j == 0 || j == 2 * n - 2)
                {
                    nozeroperow[index] -= 3;
                    nozeroperow[index1] -= 3;
                }
            }
        nozeroperow[0] -= 2;
        nozeroperow[2 * n - 2] -= 2;
        nozeroperow[(2 * n - 1) * (2 * n - 1)] -= 2;
        nozeroperow[(2 * n - 1) * (2 * n - 1) - (2 * n - 2)] -= 2;
        nozeroperow[0 + dim1] -= 2;
        nozeroperow[2 * n - 2 + dim1] -= 2;
        nozeroperow[(2 * n - 1) * (2 * n - 1) + dim1] -= 2;
        nozeroperow[(2 * n - 1) * (2 * n - 1) - (2 * n - 2) + dim1] -= 2;

        for (int i = 2 * dim1; i <= dim; ++i)
            nozeroperow[i] = 25;
        for (int i = 2; i <= n - 3; ++i)
        {
            nozeroperow[2 * dim1 + i] = 10;
            nozeroperow[dim - i] = 10;
            nozeroperow[2 * dim1 + i * (n + 1)] = 10;
            nozeroperow[2 * dim1 + (i + 1) * (n + 1) - 1] = 10;

            nozeroperow[2 * dim1 + i + (n + 1)] = 20;
            nozeroperow[dim - i - (n + 1)] = 20;
            nozeroperow[2 * dim1 + i * (n + 1) + 1] = 20;
            nozeroperow[2 * dim1 + (i + 1) * (n + 1) - 2] = 20;
        }

        nozeroperow[2 * dim1] = 4;
        nozeroperow[2 * dim1 + 1] = 8;
        nozeroperow[2 * dim1 + n + 1] = 8;
        nozeroperow[2 * dim1 + n + 2] = 16;

        nozeroperow[2 * dim1 + n] = 4;
        nozeroperow[2 * dim1 + n - 1] = 8;
        nozeroperow[2 * dim1 + n + 1 + n] = 8;
        nozeroperow[2 * dim1 + n + n] = 16;

        nozeroperow[dim] = 4;
        nozeroperow[dim - 1] = 8;
        nozeroperow[dim - n - 1] = 8;
        nozeroperow[dim - n - 2] = 16;

        nozeroperow[dim - n] = 4;
        nozeroperow[dim - n + 1] = 8;
        nozeroperow[dim - n - 1 - n] = 8;
        nozeroperow[dim - n - n] = 16;
    }
    SparsityPatter sp_stiff_matrix(dim, nozeroperow);

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
                    vindex.push_back(index[2 * (i - 1) + k, 2 * (j - 1) + o]);
                    coord.push_back(std::make_pair(k, o));
                }
            for (int x = 0; x <= 8; ++x)
                for (int y = x; y <= 8; ++y)
                    if (vindex[x] != -1 && vindex[x] != -2)
                        if (coord[y].first - coord[x].first <= 1 && coord[y].second - coord[x].second <= 1 && vindex[y] != -1 && vindex[y] != -2)
                        {
                            sp_stiff_matrix.add(vindex[x], vindex[y]);
                            sp_stiff_matrix.add(vindex[x] + dim1, vindex[y] + dim1);
                        }
            for (int x = 0; x <= 8; ++x)
                if (vindex[x] != -1 && vindex[x] != -2)
                    for (auto y : pindex)
                    {
                        sp_stiff_matrix.add(y, vindex[x]);
                        sp_stiff_matrix.add(y, vindex[x] + dim1);
                        sp_stiff_matrix.add(vindex[x], y);
                        sp_stiff_matrix.add(vindex[x] + dim1, y);
                    }
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
            double px00 = (n - 2 * rindex) / n;
            double py00 = (2 * lindex - n - 2) / n;
            pindex.push_back(2 * dim1 + i * (n + 1) + j - 1);
            double px10 = (n - 2 * rindex) / n;
            double py10 = (2 * lindex - n) / n;
            pindex.push_back(2 * dim1 + i * (n + 1) + j);
            double px11 = (n - 2 * rindex - 2) / n;
            double py11 = (2 * lindex - n) / n;
            pindex.push_back(2 * dim1 + (i - 1) * (n + 1) + j);
            double px01 = (n - 2 * rindex - 2) / n;
            double py01 = (2 * lindex - n - 2) / n;
            pindex.push_back(2 * dim1 + (i - 1) * (n + 1) + j - 1);

            pgv[0][0] = px00;
            pgv[0][1] = py00;
            pgv[1][0] = px10;
            pgv[1][1] = py10;
            pgv[2][0] = px11;
            pgv[2][1] = py11;
            pgv[3][0] = px01;
            pgv[3][1] = py01;
            plv[0][0] = arr[0][0];
            plv[0][1] = arr[0][1];
            plv[1][0] = arr[1][0];
            plv[1][1] = arr[1][1];
            plv[2][0] = arr[2][0];
            plv[2][1] = arr[2][1];
            plv[3][0] = arr[3][0];
            plv[3][1] = arr[3][1];

            std::vector<int> vindex;
            std::vector<std::pair<int, int>> coor;
            double x00 = (n - i) / n;
            double y00 = (j - 1 - n) / n;
            vindex.push_back(index[i + 1][j]);
            coor.push_back(std::make_pair(i + 1, j));
            double x10 = (n - i) / n;
            double y10 = (j - n) / n;
            vindex.push_back(index[i + 1][j + 1]);
            coor.push_back(std::make_pair(i + 1, j + 1));
            double x11 = (n - i + 1) / n;
            double y11 = (j - n) / n;
            vindex.push_back(index[i][j + 1]);
            coor.push_back(std::make_pair(i, j + 1));
            double x01 = (n - i + 1) / n;
            double y01 = (j - n - 1) / n;
            vindex.push_back(index[i][j]);
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
            auto ppoint = rectangle_coord_transform.local_to_global(q_point, plv, pgv);
            for (int l = 0; l < n_quadrature_point; l++)
            {
                double Jxy = quad_info.weight(l) * rectangle_coord_transform.local_to_global_jacobian(q_point[l], lv, gv);
                for (int base1 = 0; base1 < template_element.n_dof(); base1++)
                {
                    if (vindex[base1] != -1)
                    {
                        for (int base2 = 0; base2 < template_element.n_dof(); base2++)
                            if (vindex[base2] != -1)
                            {
                                stiff_mat.add(vindex[base1], vindex[base2], Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv)));
                                stiff_mat.add(vindex[base1] + dim1, vindex[base2] + dim1, Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv)));
                            }
                            else
                            {
                                rhs[vindex[base1]] -= ux[coor[base2].first][coor[base2].second] * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv));
                                rhs[vindex[base1] + dim1] -= uy[coor[base2].first][coor[base2].second] * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), rectangle_basis_function[base2].gradient(point[l], gv));
                            }
                    }
                    for (int base2 = 0; base2 < template_element.n_dof(); base2++)
                    {
                        if (vindex[base1] != -1)
                        {
                            double T1[2], T2[2];
                            T1[0] = rectangle_basis_function[base2].value(point[l], pgv);
                            T1[1] = 0;
                            T2[0] = 0;
                            T2[1] = T1[0];
                            stiff_mat.add(vindex[base1], pindex[base2], -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T1));
                            stiff_mat.add(pindex[base2], vindex[base1], -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T1));
                            stiff_mat.add(vindex[base1] + dim1, pindex[base2], -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T2));
                            stiff_mat.add(pindex[base2], vindex[base1] + dim1, -Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T2));
                        }
                        else
                        {
                            double T1[2], T2[2];
                            T1[0] = 1;
                            T1[1] = 0;
                            T2[0] = 0;
                            T2[1] = 1;
                            double value = rectangle_basis_function[base2].value(point[l], pgv);
                            rhs[vindex[base2]] += ux[coor[base1].first][coor[base1].second] * value * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T1);
                            rhs[vindex[base2]] += uy[coor[base1].first][coor[base1].second] * value * Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv), T2);
                        }
                    }
                }
            }
        }

    //构造稳定化矩阵C
}