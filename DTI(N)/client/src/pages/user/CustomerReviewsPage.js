import React, { useState } from 'react';
import { Modal, Form, Input, Button, notification } from 'antd';
import Layout from '../../components/Layout/Layout';
import '../../styles/review_form.css'; // Import custom CSS for the modal styling

const CustomerReviewsPage = () => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [reviews, setReviews] = useState([
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum ac diam sit amet quam vehicula elementum.",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
  ]);
  const [form] = Form.useForm();

  const showModal = () => {
    setIsModalVisible(true);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
  };

  const onFinish = (values) => {
    // Add the new review to the reviews state
    setReviews([...reviews, values.review]);
    // Close the modal
    setIsModalVisible(false);
    // Reset the form fields
    form.resetFields();
    // Show a notification that the review has been submitted
    notification.success({
      message: 'Review Submitted',
      description: 'Thank you for submitting your review!',
    });
  };

  return (
    <Layout>
      <div className="container py-7" style={{ paddingTop: '50px', padding: '4rem', marginBottom: '50px', minHeight: 'calc(100vh - 150px)' }}>
        <div className="text-center mb-5">
          <h1 className="mb-3">Customer Reviews</h1>
          <p className="lead">Read what our customers have to say about us!</p>
        </div>

        {/* Display existing reviews */}
        <div className="row">
          {reviews.map((review, index) => (
            <div className="col-lg-4 mb-4" key={index}>
              <div className="card h-100 shadow-sm">
                <div className="card-body">
                  <p>{review}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Optional: Add a button to submit a new review */}
        <div className="text-center mt-5">
          <button className="btn btn-primary btn-lg" onClick={showModal}>Submit a Review</button>
        </div>

        {/* Review form modal */}
        <Modal
          title="Submit a Review"
          visible={isModalVisible}
          onCancel={handleCancel}
          footer={null}
          className="custom-modal" // Apply custom modal class
        >
          {/* Review form */}
          <Form form={form} onFinish={onFinish} className="my-form">
            <Form.Item
              name="review"
              rules={[{ required: true, message: 'Please enter your review' }]}
            >
              <Input.TextArea rows={4} placeholder="Write your review here" className="review-input" />
            </Form.Item>

            <Form.Item>
              <div className="text-center">
                <Button type="primary" htmlType="submit">Submit Review</Button>
              </div>
            </Form.Item>
          </Form>
        </Modal>
      </div>
      <div style={{ height: '0px' }}></div> {/* Spacer to push content above the footer */}
    </Layout>
  );
};

export default CustomerReviewsPage;
