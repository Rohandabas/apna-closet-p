import React from "react";
import Layout from "./../components/Layout/Layout";

const Policy = () => {
  return (
    <Layout title={"Privacy Policy"}>
      <div className="row contactuss mt-5">
        <div className="col-md-6">
          <img
            src="/images/TermsAndConditions.jpg"
            alt="contactus"
            style={{ width: "70%" }}
          />
        </div>
        <div className="col-md-6">  
            <p className="text-justify mt-3"><h5>User Agreement:</h5>
          Users agree to abide by the terms and conditions outlined herein by using our website and services.</p>
          <p><h5>Rental Terms:</h5>
    Users must meet eligibility criteria to rent items, and rental duration, fees, and payment terms are specified.</p>
    <p className="text-justify mt-2"><h5>Item Condition and Use:</h5>
    Users agree to return rented items in the same condition as received, and prohibited uses of rented items are outlined.</p>
    <p className="text-justify mt-2"><h5>Liability and Indemnification:</h5>
    This company is not liable for damages or losses incurred by users, and users agree to indemnify this company against any claims arising from their use of the rental services.</p>
    <p className="text-justify mt-2"><h5>Intellectual Property Rights:</h5>
    All content and intellectual property displayed on the website belong to matrix company, and users are prohibited from unauthorized use</p>
        </div>
      </div>
    </Layout>
  );
  };

export default Policy;


{/* <p><h5>Rental Terms:</h5>
Users must meet eligibility criteria to rent items, and rental duration, fees, and payment terms are specified.</p>
<p><h5>Item Condition and Use:</h5>
Users agree to return rented items in the same condition as received, and prohibited uses of rented items are outlined.</p>
<p><h5>Liability and Indemnification:</h5>
This company is not liable for damages or losses incurred by users, and users agree to indemnify this company against any claims arising from their use of the rental services.</p>
<p><h5>Intellectual Property Rights:</h5>
All content and intellectual property displayed on the website belong to matrix company, and users are prohibited from unauthorized use.</p>
</div> */}


// return (
//   <Layout title={"Privacy Policy"}>
//     <div className="row-contactus">
//       <div className="col-md-6">
//         <img
//           src="/images/contactus.jpeg"
//           alt="contactus"
//           style={{ width: "100%" }}
//         />
//       </div>
//     <div className="col-md-4">
//         <p className="text-justify mt-2"><h5>User Agreement:</h5>
//         Users agree to abide by the terms and conditions outlined herein by using our website and services.</p>
//       </div>
//     </div>
//   </Layout>
// );
// };