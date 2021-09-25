import math
import pybullet as p
from pybullet_utils import bullet_client
import pybullet_data






if __name__ == '__main__':

    server = bullet_client.BulletClient(connection_mode=p.GUI_SERVER)

    print("Connecting to bullet server")
    CONNECTION_METHOD = p.SHARED_MEMORY
    client = bullet_client.BulletClient(connection_mode=CONNECTION_METHOD)

    server.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    z2y = p.getQuaternionFromEuler([0, 0, 0])
    # s_plane = server.loadURDF('plane_implicit.urdf', [0.5, 0.5, 0.5], z2y, useMaximalCoordinates=True)
    c_plane = client.loadURDF('plane_implicit.urdf', [0, 0, 0], z2y, useMaximalCoordinates=True)

    # print(s_plane)
    # print(c_plane)

    # s_state = server.saveState()
    # c_state = client.saveState()
    # print(client.getBasePositionAndOrientation(c_plane))
    # client.restoreState(s_state)
    # print(client.getBasePositionAndOrientation(0))


    while 1:
        pass


