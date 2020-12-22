using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraControl : MonoBehaviour
{
    [SerializeField]
    GameObject m_centerObject = null;

    Vector3 dv;
    // Start is called before the first frame update
    void Start()
    {
        dv = transform.position - m_centerObject.transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = m_centerObject.transform.position + dv;
    }
}
