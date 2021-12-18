def render_output(point,tri,N):
    # Create POV-Ray include file with a 3D curve in it, made of cylinders and
    # mesh
    f=open("render_mesh_{step}.inc".format(step = N),"w")

    # Loop over all triangles in the mesh

    for i in range(len(tri)):
        (x_1,y_1,z_1)=point[tri[i][0]]
        (x_2,y_2,z_2)=point[tri[i][1]]
        (x_3,y_3,z_3)=point[tri[i][2]]

        # Print a sphere
        f.write("triangle{<%.4f,%.4f,%.4f>,<%.4f,%.4f,%.4f>,<%.4f,%.4f,%.4f>}\n"%(x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3))

    f.close()
    return
