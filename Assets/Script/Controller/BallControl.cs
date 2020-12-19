using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallControl : MonoBehaviour
{
    // Start is called before the first frame update
    Rigidbody m_rigidbody;
    void Start()
    {
        m_rigidbody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        float x = Input.GetAxisRaw("Horizontal");
        float z = Input.GetAxisRaw("Vertical");
        float y = 0;
        if(Input.GetButton("Jump"))
        {
            y = 5.0f * m_rigidbody.mass;
        }
        if(Mathf.Abs(x) > 1e-5f || Mathf.Abs(z) > 1e-5f || y > 1e-5f)
        {
            m_rigidbody.AddForce(x, y, z);
        }
    }
}
