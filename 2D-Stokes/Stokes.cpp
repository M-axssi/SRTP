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

    std::vector<unsigned int> nozeroperow(dim);
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

    SparsityPatter sp_stiff_matrix(dim, nozeroperow);

    for (int i = 0; i <= n * n - 1; ++i)
    {
        int index1 = i;
        int index2 = n * n - 1 + i;
        sp_stiff_matrix.add(index1, index1);
        sp_stiff_matrix.add(index2, index2);

        int rindex = i / n;
        int lindex = i - i / n * n;
        std::vector<int> index;
        index.push_back(2 * n * n - 1 + rindex * (n + 1) + lindex + 1);
        index.push_back(2 * n * n - 1 + rindex * (n + 1) + lindex + 2);
        index.push_back(2 * n * n - 1 + (rindex + 1) * (n + 1) + lindex + 1);
        index.push_back(2 * n * n - 1 + (rindex + 1) * (n + 1) + lindex + 2);
        for (auto x : index)
        {
            for (auto y : index)
                sp_stiff_matrix.add(x, y);
            sp_stiff_matrix.add(index1, x);
            sp_stiff_matrix.add(index2, x);
            sp_stiff_matrix.add(x, index1);
            sp_stiff_matrix.add(x, index2);
        }
    }
    sp_stiff_matrix.compress();
    SparseMatrix<double> stiff_mat(sp_stiff_matrix);

    //构造A
    for (int i = 0; i <= n * n - 1; ++i)
    {
        int index1 = i;
        int index2 = n * n - 1 + i;
    }
}